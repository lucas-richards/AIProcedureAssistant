# AI Procedure Assistant

AI Procedure Assistant is an internal knowledge assistant that answers procedure questions using only your organization’s procedure documents.

## Purpose

This app helps teams quickly find accurate, step-by-step answers from internal procedure files without relying on outside knowledge.

It is designed to:
- Search your procedure documents in `procedures/`
- Return concise, procedure-grounded answers
- Show supporting source snippets (chapter/page/quote) when available
- Respond with a strict fallback when information is not present in the procedures

## How it works

- Backend: FastAPI app in `backend/app.py`
- UI: Chat interface served from `backend/static/chat.html`
- Retrieval: LlamaIndex RAG pipeline over files in `procedures/`
- LLM + embeddings: Ollama (`llama3.1:latest` and `nomic-embed-text`)

On startup, the app indexes documents from `procedures/` and exposes a chat endpoint.

## Routes

- `GET /` → redirects to chat page
- `GET /chat` → serves the chat UI
- `POST /chat` → handles chat requests

## Prerequisites

- Linux/macOS shell
- Python virtual environment at `venv/`
- Ollama running locally at `http://127.0.0.1:11434`
- Required Ollama models pulled:
  - `llama3.1:latest`
  - `nomic-embed-text`

## Run the app

From the repository root:

```bash
source venv/bin/activate && cd backend && uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open:

- `http://localhost:8000`


