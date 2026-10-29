# FinSight AI
### Local-LLM Financial Analytics, RAG Chatbot and Tool-Calling Agent

FinSight AI is a Python application that analyses bank transactions stored in
PostgreSQL and adds three AI features on top, all running locally through
Ollama with no paid APIs:

1. **Analytics dashboard** – KPIs, charts, transaction explorer and AI spending insights
2. **Document chatbot (RAG)** – answers personal-finance questions from a knowledge base, with citations
3. **AI agent** – answers questions about the transactions by calling safe, read-only database tools

Every AI output that contains money amounts is checked against the database or the
source documents before it is shown.

> **Data notice:** The project uses **synthetic** transaction data made by its
> own generators. Do not upload real bank statements to a public deployment.

---

## 1. Features

### Analytics Dashboard (`app.py`)
- KPI cards (income, expenses, net cash flow, savings rate) with change vs the previous period
- Sidebar filters: user, date range, category, transaction type
- Weekly spending trend, category donut, top merchants, income vs expenses
- Transaction explorer with search, sorting, pagination and CSV export
- Statement upload (CSV / Excel) with bank-format detection (HDFC, SBI, ICICI),
  cleaning, validation, duplicate-file check and dry-run preview before loading
- AI spending report from Gemma 3, with a **rule-based fallback** when Ollama is
  offline or the AI output fails amount validation
- Light and dark theme support

### Document Chatbot – RAG (`rag/`)
- Loads `.txt` documents from `rag/knowledge_base/`
- Word-based chunking (120 words, 25 overlap)
- Embeddings with `nomic-embed-text` through Ollama
- In-memory NumPy vector store with cosine similarity
- Top-3 retrieval with a 0.25 minimum-similarity threshold
- Structured JSON answers validated with **Pydantic**
- Citation validation (the answer may only cite retrieved sources)
- Money-amount validation against the retrieved text
- Says "not enough information" instead of guessing when the context is missing

### AI Agent with Tool Calling (`agent/`)
- Gemma 3 picks one tool and its arguments as **constrained JSON**
  (Ollama `format: "json"`), because Gemma 3 has no native function calling in Ollama
- 6 allowlisted, read-only PostgreSQL tools:
  `get_total_expenses`, `get_top_expenses`, `get_category_spending`,
  `get_monthly_spending`, `get_top_merchants`, `search_transactions`
- Safety layers:
  - Only allowlisted tools; unknown tools and extra arguments are rejected
  - Arguments validated (limit 1–20, search text length, wildcard escaping)
  - Parameterised SQL only (`psycopg.sql`), no string-built queries
  - The **user scope is set by the app**, never by the model
  - Database error details are logged, not shown to the user
- **Answer validation:** every amount in the model's answer must exist in the
  tool result; otherwise the answer is rebuilt from the database
- Still answers from the database if the model goes down after the query
- "View agent execution" panel shows the tool, scope, timings and raw result

### REST API (`api/main.py`)
- FastAPI with `GET /health` and `POST /ask` (RAG question answering)
- Pydantic request and response models

---

## 2. Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| UI | Streamlit |
| API | FastAPI, Uvicorn |
| Database | PostgreSQL (psycopg 3) |
| Data processing | Pandas, NumPy |
| Charts | Plotly |
| Local LLM runtime | Ollama |
| Language model | `gemma3:4b` |
| Embedding model | `nomic-embed-text` |
| Vector search | In-memory NumPy cosine similarity |
| Output validation | Pydantic, custom amount and citation checks |
| Testing | pytest, unittest.mock |
| Configuration | python-dotenv |

---

## 3. Architecture

```text
                         Streamlit App (app.py)
                                  |
      +-------------+-------------+-------------+--------------+
      |             |                           |              |
  Dashboard     Document Chatbot            AI Agent     RAG Evaluation
      |             |                           |          Dashboard
 PostgreSQL     Knowledge base (.txt)     Question
      |             |                           |
 Pandas KPIs    Chunk -> Embed (Ollama)   Gemma 3 picks tool (JSON)
 + charts           |                           |
      |         NumPy vector store        Validate tool + arguments
 Gemma 3            |                           |
 insights       Top-3 retrieval           Read-only SQL tool
      |             |                     (user scope from app)
 Amount check   Gemma 3 answer (JSON)           |
      |             |                     Gemma 3 writes answer
 Rule-based     Pydantic + citation             |
 fallback       + amount checks           Amount check vs DB result
                    |                           |
                Answer + sources          Answer (or DB template)

             FastAPI (api/main.py): /health, /ask -> RAG pipeline
```

---

## 4. Project Structure

```text
Finsight_AI/
|-- app.py                     # Streamlit app (dashboard + navigation)
|-- requirements.txt
|-- .env.example
|-- .streamlit/config.toml     # Theme settings
|
|-- agent/
|   |-- financial_agent.py     # Tool selection, validation, answer check
|   |-- tools.py               # 6 read-only PostgreSQL tools
|   |-- agent_ui.py            # Agent chat page
|
|-- rag/
|   |-- document_loader.py     # Loading and chunking
|   |-- embeddings.py          # Ollama embeddings
|   |-- vector_store.py        # NumPy cosine-similarity search
|   |-- rag_pipeline.py        # Retrieval, prompt, Pydantic, validation
|   |-- chat_ui.py             # Chatbot page
|   |-- knowledge_base/        # Source documents
|
|-- ai/llm_insights.py         # AI spending report + fallback
|-- analytics/                 # KPI and SQL query helpers
|-- api/main.py                # FastAPI endpoints
|-- db/                        # Connection and transaction loading
|-- etl/                       # Bank detection, cleaning, mapping, validation, loading
|-- generators/                # Synthetic users, accounts and transactions
|-- modules/                   # Statement upload page, config loader
|-- config/                    # Categories, merchants, banks, personas (JSON)
|-- sql/schema.sql             # Database schema
|-- evaluation/                # Evaluators, test sets, results, dashboard
|-- tests/                     # pytest unit tests + agent evaluation scripts
|-- docs/PROJECT_ARCHITECTURE.md
```

---

## 5. Installation (Windows PowerShell)

### Prerequisites
- Python 3.11
- PostgreSQL
- Ollama
- Git

### Step 1: Clone
```powershell
git clone https://github.com/YOUR-USERNAME/finsight-ai.git
cd finsight-ai
```

### Step 2: Virtual environment and packages
```powershell
py -3.11 -m venv .venv311
.venv311\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Step 3: Environment variables
```powershell
Copy-Item .env.example .env
```
Edit `.env`:
```dotenv
DB_HOST=localhost
DB_PORT=5432
DB_NAME=finsight_ai
DB_USER=postgres
DB_PASSWORD=your_database_password

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=gemma3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=180
```
Never commit `.env`.

### Step 4: Ollama models
```powershell
ollama pull gemma3:4b
ollama pull nomic-embed-text
ollama list
```

### Step 5: Database
```powershell
psql -U postgres -c "CREATE DATABASE finsight_ai;"
psql -U postgres -d finsight_ai -f sql/schema.sql
```
Then load synthetic data by uploading a CSV from `data/` in the app's **Upload** tab.

### Step 6: Run
```powershell
python -m streamlit run app.py          # Web app
uvicorn api.main:app --reload           # API (docs at http://127.0.0.1:8000/docs)
```

---

## 6. Testing and Evaluation

### Unit tests (no Ollama or database needed – mocked)
```powershell
python -m pytest
```
**Result: 39 passed.** They cover tool validation, rejection of unknown and unsafe
tools, SQL user filtering, invented-amount replacement, friendly errors when
Ollama is offline, and RAG pipeline validation.

### Recorded evaluation results (local run, `gemma3:4b`, October 2026)

| Evaluation | Questions | Result |
|------------|-----------|--------|
| Agent tool selection | 18 | 18/18 (100%) |
| Agent on unseen phrasings (tool + arguments) | 30 | 30/30 (100%), 13.7 s avg |
| Agent safety (unsafe requests rejected) | 24 | 24/24 (100%) |
| Agent financial answers match the database | 6 | 6/6 (100%), 27–35 s avg |
| RAG (25-question test set) | 25 | 25/25 executed |

**RAG metrics (25 questions)**

| Metric | Result |
|--------|--------|
| Retrieval Hit@1 / Hit@3 / MRR | 1.00 / 1.00 / 1.00 |
| Citation validity | 100% |
| Abstention on unsupported questions | 5/5 |
| Hallucination-trap questions passed | 5/5 |
| Grounding (LLM judge, 15 eligible questions) | 1.00, 0 hallucination flags |
| Average response time | 18.7 s (CPU, local) |

Run them yourself:
```powershell
python -m evaluation.rag_evaluator
python -m tests.evaluate_agent
python -m tests.evaluate_agent_unseen
python -m tests.evaluate_agent_safety
python -m tests.evaluate_financial_answers
```

### Honest reading of these results
- The RAG knowledge base has **only 3 documents**, so perfect retrieval is
  expected and does not prove it would scale.
- Test sets are small and written by the author.
- Grounding scores come from an LLM judge using the same model family; this is an
  estimate, not proof.
- Results can change with other models, hardware or documents.

---

## 7. Known Limitations
- Embeddings are kept in memory and rebuilt at startup (no vector database).
- The knowledge base is small (3 documents, `.txt` only).
- Responses take 15–35 s on CPU because the models run locally.
- The agent answers one tool call per question; it cannot chain several tools.
- No login: user scope is selected in the app, which is fine for a demo but not for real users.
- Runs locally only; it is not deployed to a server.
- Demo data is synthetic; a few generated merchants are "Unknown".

## 8. Possible Next Steps
- Persistent vector storage (pgvector or FAISS) and a larger knowledge base
- Multi-step tool calling
- Authentication with per-user data access
- Docker packaging and cloud deployment
- Monitoring of latency and failure rates

---

## 9. What This Project Demonstrates
- Python application design across ETL, analytics, AI and API layers
- SQL and PostgreSQL with parameterised queries
- LLM integration with a local model (Ollama)
- Retrieval-Augmented Generation with citations
- Tool calling for a model without native function support
- Guardrails: allowlists, argument validation, output validation, fallbacks
- Evaluation design (retrieval metrics, safety tests, answer accuracy)
- Automated testing with mocks
- Streamlit and FastAPI development

---

## 10. Disclaimer
FinSight AI is an educational portfolio project. Its outputs are not financial,
tax, investment or legal advice. Use synthetic or properly authorised data only.
