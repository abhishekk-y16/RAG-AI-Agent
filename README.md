# RAG AI Agent

An End‑to‑End Document Intelligence app that Ingests files, performs semantic search, and answers questions with Groq‑powered RAG—wrapped in a clean React dashboard.

## Highlights 

- Multi‑document Ingestion and semantic retrieval
- Intent‑aware tool routing for smarter responses
- Chat, Documents,Quick Tools, Workflows, and Analytics UI

**Tech Stack:** FastAPI  + Uvicorn, Groq, React + Vite

## Quick Start

### Backend

1. Set your Groq key:
   ```bash
   setx GROQ_API_KEY "YOUR_GROQ_API_KEY" 
   ```
2. Install and run:
   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the app at `http://localhost:5173`.

## First Use

Ingest the sample knowledge base once, then start chatting:

```bash
curl -X POST http://localhost:8000/ingest
```

Example chat request:

```bash
curl -X POST http://localhost:8000/chat \
   -H "Content-Type: application/json" \
   -d "{\"message\": \"Summarize the document\"}"
```
