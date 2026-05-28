
---

```markdown
# 🏢 Enterprise AI Knowledge Assistant ( RAG Pipeline)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-v0.2-green.svg)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Memory-336791.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black.svg)
![Ragas](https://img.shields.io/badge/Ragas-Evaluation-orange.svg)

## 📌 Project Overview
**Enterprise AI Knowledge Assistant** is an enterprise-grade RAG (Retrieval-Augmented Generation) system designed to help internal employees query information from the company handbook (Basecamp Employee Handbook). 

What sets this project apart from basic RAG applications is the integration of **RBAC (Role-Based Access Control)** at the Vector database level, combined with **Cross-Encoder Reranking** and an **Automated Evaluation Pipeline** using the Ragas framework. The core system is optimized to run entirely locally, ensuring 100% data privacy for internal corporate data.

## ✨ Key Features
- 🔒 **Role-Based Access Control (RBAC Metadata Filtering):** Automatically blocks access to sensitive documents (e.g., salary, bonuses, severance) for unauthorized roles (e.g., Interns) directly at the Qdrant Vector search layer, strictly preventing Data Leakage.
- 🎯 **High-Precision Retrieval (Advanced Retrieval):** Combines Vector Search and a Cross-Encoder model (`BAAI/bge-reranker-base`) to score and rerank context, providing the LLM with only the most highly relevant information.
- 🧠 **Stateful Memory:** Stores the entire chat history in PostgreSQL, allowing users to ask follow-up questions naturally with full context awareness.
- 📊 **Automated Evaluation (Evaluation-Driven):** Integrates the Ragas framework using *LLM-as-a-judge* (Llama-3.1-70B via Groq API) to quantitatively measure Faithfulness and Answer Relevancy.
- 💻 **Privacy & Cost Optimization:** Utilizes Ollama (Llama 3.2) and HuggingFace Embeddings to process 100% of text generation locally on the machine, incurring zero API costs for the core chat pipeline.

## 🏗️ System Architecture

1. **Ingestion Pipeline:** Read PDF -> Chunking (`RecursiveCharacterTextSplitter`) -> Vector Embedding (HuggingFace BGE) -> Inject Metadata (Role, Source) -> Qdrant.
2. **Retrieval Pipeline:** User Query -> Retrieve Chat History (Postgres) -> Qdrant (RBAC Filter) -> Fetch Top 20 -> Reranker (Cross-Encoder) -> Fetch Top 4.
3. **Generation Pipeline:** Prompt Formulation (Mandatory citations, Anti-Hallucination) -> Llama 3.2 (Ollama) -> Generate Response -> Save to Postgres.

## 📂 Project Structure
```text
├── app/
│   ├── core/              # Environment variables, configs
│   ├── models/            # SQLAlchemy schemas (User, Conversation)
│   ├── services/
│   │   ├── ingestion/     # PDF processing, Chunking logic
│   │   ├── retrieval/     # Qdrant Search, Vector Store setup
│   │   ├── generation/    # Prompt Templates, Reranker module
│   │   └── workflow/      # Main RAG Pipeline, Memory management
├── data/
│   └── handbook/          # PDF data categorized by access level (public, manager, admin)
├── evaluation/
│   ├── ground_truth.json  # Benchmark dataset for evaluation
│   └── run_ragas.py       # Automated evaluation script using Ragas & Groq
├── frontend/
│   └── run_app.py             # Streamlit Web UI
├── docker-compose.yml     # Infrastructure setup (Qdrant & PostgreSQL)
├── requirements.txt       # Dependency management (Version-locked)
├── run_ingest.py          # Script to populate the Vector DB
└── run_chat.py            # CLI chat testing script

```

## 🚀 Getting Started

### 1. Prerequisites

* Docker & Docker Compose
* Python 3.10+
* [Ollama](https://ollama.com/) (Pull the model: `ollama run llama3.2`)

### 2. Initialize Database Infrastructure

Open your terminal and run the following command to start Qdrant and PostgreSQL:

```bash
docker-compose up -d

```

### 3. Install Dependencies

Use the provided `requirements.txt` to ensure a stable LangChain v0.2 ecosystem that is fully compatible with Ragas:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add the following configurations:

```env
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=knowledge_base
SQLALCHEMY_DATABASE_URI=postgresql://postgres:postgres@localhost:5432/knowledge_base
QDRANT_URL=http://localhost:6333

# Evaluation API Keys (Optional, for the evaluation pipeline only)
GROQ_API_KEY=your_groq_key_here

```

### 5. Data Ingestion

Process all PDFs in the `data/handbook/` directory, embed them, and push them to Qdrant with RBAC Metadata attached:

```bash
python run_ingest.py

```

### 6. Launch Web UI

```bash
streamlit run frontend/app.py

```

> Access `http://localhost:8501` to test the application. You can toggle your Role (Intern, Manager, Admin) in the Sidebar to test the RBAC functionality and observe how sensitive documents are dynamically blocked.

## 📈 Model Evaluation

This system utilizes the **Ragas framework** paired with the `llama-3.1-70b-versatile` judge (via Groq API) to evaluate accuracy against an internal Ground Truth dataset.

Run the evaluation pipeline:

```bash
python evaluation/run_ragas.py

```

*Average Results (based on `rag_evaluation_report.csv`):*

* **Faithfulness (Anti-Hallucination):** > `0.90` (Almost all answers are 100% grounded in the retrieved documents).
* **Answer Relevancy:** > `0.85` (Responses are highly relevant and directly address the user's specific questions without drifting).

## 🔮 Future Enhancements

* Integrate Data Connectors for real-time syncing with Google Drive, Notion, or Slack.
* Upgrade the chunking mechanism to Semantic Chunking to better preserve the context of complex financial tables.
* Abstract the core logic into a REST API using FastAPI to completely decouple the frontend and backend.


