
---

````md
# 🧠 Member QA — Deep-Learning Question Answering API

A **deep-learning powered API** that answers **natural-language questions about members** using only information from the public **`/messages`** API — **no external LLM APIs required.**

---

### 🚀 Example
Ask:
> _“When is Layla planning her trip to London?”_

API responds:
```json
{ "answer": "Layla has a chauffeur scheduled to pick her up in London on March 19, 2025 at 9:30 AM." }
````

---

## 📌 Overview

This service allows clients to ask free-form natural-language questions and get precise structured answers.

| Example Question                             | Expected Answer Type |
| -------------------------------------------- | -------------------- |
| “When is Layla planning her trip to London?” | Date / Schedule      |
| “How many cars does Vikram Desai have?”      | Number / Count       |
| “What are Amira’s favorite restaurants?”     | Preferences / List   |

The system pulls messages, builds embeddings, performs semantic search, and extracts the answer — all locally.

---

## ⚙️ Architecture

The QA pipeline uses **Retrieval-Augmented Deep Learning**:

1. Fetch messages from the public `/messages` API
2. Store them in `messages.json`
3. Generate embeddings using:
   `sentence-transformers/all-MiniLM-L6-v2`
4. Detect which **member** the question refers to
5. Retrieve only messages **from that member**
6. Run an extractive QA model:
   `deepset/roberta-base-squad2`
7. If no exact span is found → fallback to the best matching message

### 🔐 Data Guarantee

> If the question is about **Amira**, only **Amira’s messages** are used — never from other members.

---

## 🔥 Features

| Capability                      | Status |
| ------------------------------- | ------ |
| Natural-language question input | ✅      |
| Semantic vector search          | ✅      |
| Extractive QA model             | ✅      |
| Member-restricted retrieval     | ✅      |
| Best-effort fallback            | ✅      |
| `/refresh` endpoint             | ✅      |
| Built-in Web UI (`index.html`)  | ✅      |

---

## 🧠 Tech Stack

| Component                | Role               |
| ------------------------ | ------------------ |
| FastAPI                  | REST API           |
| Sentence-Transformers    | Message embeddings |
| HuggingFace Transformers | QA inference       |
| Torch                    | Model runtime      |
| NumPy                    | Cosine similarity  |
| HTML + Vanilla JS        | Browser UI         |

---

## 🔌 API Endpoints

| Method | Route               | Description                          |
| ------ | ------------------- | ------------------------------------ |
| `GET`  | `/ask?question=...` | Ask any natural-language question    |
| `GET`  | `/refresh`          | Reload messages + rebuild embeddings |
| `GET`  | `/`                 | Web UI for manual testing            |

### Sample Request

```
GET /ask?question=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F
```

### Sample Response

```json
{ "answer": "Layla has a chauffeur scheduled to pick her up in London on March 19, 2025 at 9:30 AM." }
```

---

## ▶️ Running Locally

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Start the server

```
uvicorn main:app --host 0.0.0.0 --port 8080
```

### 3️⃣ Open the UI

```
http://localhost:8080
```

---

## 🧪 Example Questions to Try

| Question                               |                        |
| -------------------------------------- | 
| “What is Layla’s seating preference?”  |
| “When is Amira traveling to Tokyo?”    | 
| “How many cars does Vikram Desai own?” |
| “Book me a restaurant”                 | 

---

## 📘 Bonus 1 — Design Notes

Alternative approaches evaluated:

| Approach                   | Result / Limitation                                 |
| -------------------------- | --------------------------------------------------- |
| Keyword Search             | Too brittle — fails on phrasing changes             |
| TF-IDF + Cosine Similarity | Higher recall but member confusion                  |
| End-to-End LLM             | Best performance but requires cloud GPU + API costs |

### ✔ Final Choice — Hybrid Retrieval + Extractive QA

Combines:

* **Embeddings** → member-filtered semantic retrieval
* **RoBERTa QA** → exact answer spans
* **Fallbacks** → prevents blank / hallucinated responses

---

## 📘 Bonus 2 — Data Insights

| Category      | Observation                                          | Impact                           |
| ------------- | ---------------------------------------------------- | -------------------------------- |
| Identity      | Slight name variations across messages               | Splits persona unless normalized |
| Content       | Some entries lack message text                       | Must be discarded                |
| Time          | Many events dated 2025–2026                          | Indicates simulated dataset      |
| Semantics     | Some facts implied but not explicit                  | Fallback required                |
| Topic overlap | Heavy overlap in travel / restaurants across members | Naive search confuses personas   |

These observations directly shaped the final solution.

---

## 📂 Project Structure

```
project/
│ main.py
│ index.html
│ requirements.txt
│ README.md
│ messages.json      (autogenerated)
└── models/          (optional — cached downloads)
```

---

## 🚀 Deployment

Works on:

| Platform         | Supported |
| ---------------- | --------- |
| Local machine    | ✅         |
| Docker container | ✅         |
| Render           | ✅         |
| Railway          | ✅         |
| Google Cloud Run | ✅         |

Models download once and remain cached.

---

## 📫 Support & Extensions

Open a GitHub Issue if you'd like help adding:

* Vector database (Pinecone / FAISS / Qdrant)
* Conversation memory
* Async inference batching
* CI/CD deployment to Render
* Monitoring & observability

Always happy to assist 🤝

---

### 🏁 Final Remark

> This project demonstrates that high-quality natural-language question answering — without external LLM APIs — is possible using a carefully engineered combination of semantic retrieval, extractive QA, and strict member-level filtering.

```

---


```
