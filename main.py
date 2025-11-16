import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple

import requests
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

MESSAGES_URL = "https://november7-730026606190.europe-west1.run.app/messages"
DATA_FILE = "messages.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_NAME = "deepset/roberta-base-squad2"

app = FastAPI(
    title="Member Question-Answering Service",
    description="Answers natural-language questions using deep learning and NLP over /messages data.",
    version="1.0.0",
)


# ----------------------------------------------------------------------
# Deep Learning QA System
# ----------------------------------------------------------------------

class DeepQASystem:
    """
    Deep-learning QA system over member messages.

    - Fetches all messages from the /messages API
    - Stores raw JSON in messages.json
    - Builds embeddings for each message with a sentence-transformer
    - Uses a QA model to extract answers from the most relevant messages
    - Respects the member name: if the question is about Amira, only Amira’s
      messages are used; it will never answer with Layla’s messages.
    """

    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []
        self.docs: List[str] = []                 # text representation per message
        self.embeddings: Optional[np.ndarray] = None

        # Load models once
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.qa_pipeline = pipeline(
            "question-answering",
            model=QA_MODEL_NAME,
            tokenizer=QA_MODEL_NAME,
        )

    # ---------------- Data handling ---------------- #

    def fetch_messages_from_api(self) -> List[Dict[str, Any]]:
        """
        Retrieve messages from the public /messages endpoint.
        """
        try:
            resp = requests.get(MESSAGES_URL, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch messages from upstream API: {e}",
            )

        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("messages", "results", "items", "data"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return []

    def save_messages_to_file(self, messages: List[Dict[str, Any]]) -> None:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    def load_messages_from_file(self) -> Optional[List[Dict[str, Any]]]:
        if not os.path.exists(DATA_FILE):
            return None
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            return None
        return None

    # ---------------- Message text construction ---------------- #

    @staticmethod
    def get_member_name(msg: Dict[str, Any]) -> str:
        """
        Heuristic extraction of member name.
        Adjust field names here if your schema is different.
        """
        for key in ("member_name", "name", "user_name", "username"):
            v = msg.get(key)
            if isinstance(v, str):
                return v

        member_obj = msg.get("member")
        if isinstance(member_obj, dict):
            for key in ("name", "full_name", "display_name"):
                v = member_obj.get(key)
                if isinstance(v, str):
                    return v

        return "Unknown member"

    @staticmethod
    def get_message_body(msg: Dict[str, Any]) -> str:
        """
        Prefer explicit text fields; otherwise concatenate all string values.
        """
        for key in ("text", "message", "content", "body", "note"):
            v = msg.get(key)
            if isinstance(v, str):
                return v

        parts: List[str] = []
        for v in msg.values():
            if isinstance(v, str):
                parts.append(v)
        return " ".join(parts) if parts else str(msg)

    def build_doc_text(self, msg: Dict[str, Any]) -> str:
        """
        Build the text that will be embedded for semantic search.
        """
        member = self.get_member_name(msg)
        body = self.get_message_body(msg)
        return f"{member}: {body}"

    # ---------------- Index building ---------------- #

    def rebuild_index(self) -> None:
        """
        Load messages from file if available; otherwise fetch from API.
        Then compute sentence embeddings for each message.
        """
        cached = self.load_messages_from_file()
        if cached is None:
            messages = self.fetch_messages_from_api()
            self.save_messages_to_file(messages)
        else:
            messages = cached

        self.messages = messages
        self.docs = [self.build_doc_text(m) for m in messages]

        if not self.docs:
            self.embeddings = None
            return

        # Compute embeddings (deep learning)
        self.embeddings = self.embedder.encode(
            self.docs,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity == dot product
        )

    # ---------------- Member name extraction ---------------- #

    @staticmethod
    def extract_member_name_from_question(question: str) -> Optional[str]:
        """
        Extract a likely member name from the question.
        Very simple: look for capitalized words/phrases, skip question words.
        """
        candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", question)
        if not candidates:
            return None

        skip = {"What", "When", "How", "Who", "Where", "Which", "Why"}
        for cand in candidates:
            first = cand.split()[0]
            if first in skip:
                continue
            return cand.strip()

        return None

    # ---------------- Retrieval with optional member filter ---------------- #

    def retrieve_top_k(
        self,
        question: str,
        k: int = 5,
        person: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve indices and similarity scores of top-k most relevant messages.

        If 'person' is given, restrict to docs that mention that member name.
        """
        if self.embeddings is None or not len(self.docs):
            return []

        q_emb = self.embedder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # If a member name is detected, restrict to that member's docs
        if person:
            person_lower = person.lower()
            candidate_indices = [
                i for i, doc in enumerate(self.docs)
                if person_lower in doc.lower()
            ]
        else:
            candidate_indices = list(range(len(self.docs)))

        if not candidate_indices:
            return []

        # Build sub-matrix for those candidate docs
        sub_emb = self.embeddings[candidate_indices]
        scores = np.dot(sub_emb, q_emb)

        k = min(k, len(candidate_indices))
        top_positions = np.argsort(scores)[::-1][:k]
        top_hits = [
            (candidate_indices[pos], float(scores[pos]))
            for pos in top_positions
        ]
        return top_hits

    # ---------------- QA answering ---------------- #

    def answer(self, question: str) -> str:
        """
        Answer any natural-language question based on the member messages.

        Rules:
        - If the question mentions a member (e.g. "Amira"), only that
          member's messages are considered.
        - Use embeddings to get top-k relevant messages.
        - Use a QA model to extract a short answer span.
        - If that fails, fall back to showing the most relevant message
          from that member (or globally, if no member is detected).
        """
        if self.embeddings is None or not self.docs:
            return "I couldn't find an answer in the member messages."

        # 1) Detect member in the question
        person = self.extract_member_name_from_question(question)

        # 2) Retrieve candidates (restricted by person if available)
        top_hits = self.retrieve_top_k(question, k=5, person=person)
        if not top_hits:
            if person:
                return f"I couldn't find any messages for {person}."
            return "I couldn't find an answer in the member messages."

        best_answer: Optional[str] = None
        best_score: float = -1.0

        # 3) Try to extract a precise answer from each top context
        for idx, sim in top_hits:
            context = self.docs[idx]

            try:
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    top_k=1,
                )
            except Exception:
                continue

            # pipeline may return a dict or list
            if isinstance(result, list) and result:
                result = result[0]

            ans_text = (result.get("answer") or "").strip()
            qa_score = float(result.get("score", 0.0))

            if not ans_text:
                continue

            # Combine QA confidence with retrieval similarity
            combined = qa_score * 0.7 + sim * 0.3

            if combined > best_score:
                best_score = combined
                best_answer = ans_text

        # 4) If we have a confident, non-empty answer, return it
        if best_answer and best_score > 0.2:
            return best_answer

        # 5) Fallback: no strong span found.
        # Return the most relevant message, but only from that person if specified.
        top_idx, _ = top_hits[0]
        top_context = self.docs[top_idx]

        if person:
            return (
                f"I couldn't extract a precise answer from {person}'s messages, "
                f"but the most relevant message is: {top_context}"
            )

        return (
            "I couldn't extract a precise answer, but the most relevant message is: "
            + top_context
        )


# Global QA system instance
qa_system = DeepQASystem()


# ----------------------------------------------------------------------
# Application lifecycle
# ----------------------------------------------------------------------

@app.on_event("startup")
def on_startup() -> None:
    """
    Build the initial index when the service starts.
    """
    qa_system.rebuild_index()


# ----------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------

@app.get("/ask")
def ask(question: str = Query(..., description="Natural-language question")) -> Dict[str, str]:
    """
    Main QA endpoint.

    Example:
      GET /ask?question=When+is+Layla+planning+her+trip+to+London%3F

    Response:
      { "answer": "..." }
    """
    answer = qa_system.answer(question)
    return {"answer": answer}


@app.get("/refresh")
def refresh() -> Dict[str, str]:
    """
    Force a full reload of messages from the upstream API and rebuild the index.
    """
    qa_system.rebuild_index()
    return {"status": "ok", "message": "Messages reloaded and index rebuilt."}


@app.get("/")
def home() -> FileResponse:
    """
    Serve a simple HTML page for manual testing (index.html).
    """
    return FileResponse("index.html")
