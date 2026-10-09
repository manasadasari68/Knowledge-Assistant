# Utility Knowledge Assistant

Starter monorepo for a utility knowledge assistant, with a FastAPI API in `app/api` and a Next.js frontend in `web`.

## Prerequisites

- Python 3.10+
- Node.js 20+
- npm

## Set up PostgreSQL and the API

Use PostgreSQL with the [pgvector extension](https://github.com/pgvector/pgvector) installed. Set `DATABASE_URL` in `app/api/.env` using the `postgresql+psycopg://` format shown in `app/api/.env.example`. The configured database must already exist, and the database role must be allowed to create the `vector` extension on its first migration.

```powershell
cd app/api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env # skip if app/api/.env already exists
alembic upgrade head
uvicorn app.main:app --reload
```

The API listens on `http://localhost:8000`; its health endpoint is `http://localhost:8000/health`.

## Ingest knowledge PDFs

With the API environment active and the database migration applied, run from `app/api`:

```powershell
python -m app.services.ingest
```

This recursively processes PDFs under the repository's `knowledge/` directory. To use another directory, pass `--directory "C:\path\to\pdfs"`. Re-running ingestion updates existing documents and replaces their chunks. The ingestion foundation extracts selectable PDF text, groups nearby paragraphs into chunks, and records source filenames and page ranges. Scanned image-only PDFs need OCR, which is not included.

## Generate embeddings and try retrieval

Set `EMBEDDING_API_KEY` in `app/api/.env` (or `OPENROUTER_API_KEY` if using OpenRouter). The default embedding endpoint and model are OpenRouter's OpenAI-compatible API and `openai/text-embedding-3-small`; change `EMBEDDING_API_BASE_URL` and `EMBEDDING_MODEL` to use another compatible service/model. Set `EMBEDDING_DIMENSIONS` only if the provider supports requesting a fixed dimension. The chosen model must support text embeddings.

After ingesting or changing documents, create vectors for chunks using:

```powershell
cd app/api
python -m app.services.embed_documents
```

This embeds chunks that are missing vectors or were created by a different configured model. Use `--force` to re-embed every stored chunk. Configure the same model for indexing and question retrieval. The API's test retrieval endpoint accepts a question and optional `top_k` (1-20):

```text
GET  http://localhost:8000/api/retrieval/test?question=How%20do%20I%20replace%20a%20meter&top_k=5
POST http://localhost:8000/api/retrieval/test
```

POST JSON body: `{"question":"How do I replace a meter?","top_k":5}`. Responses contain matching chunk text, source document and chunk metadata, and cosine similarity scores. This endpoint only retrieves context; it does not generate an answer.

Chat retrieval is configured separately in `app/api/.env`: `RETRIEVAL_TOP_K` controls the maximum number of ranked chunks passed to answer generation (default `8`, maximum `20`), `RETRIEVAL_MIN_SIMILARITY` filters weak semantic matches (default `0.25`), and `RETRIEVAL_MIN_KEYWORD_SCORE` controls lexical candidate eligibility (default `0.10`). Semantic and PostgreSQL English full-text ranks are combined with reciprocal-rank fusion. If no chunks meet either relevance floor, chat keeps its existing not-enough-information response.

## Ask a knowledge-base question

Set `LLM_MODEL` in `app/api/.env` to an OpenRouter chat model. The default is `openai/gpt-4o-mini`. Send a question to `POST /api/chat`:

```json
{"question":"What should I do if my meter is faulty?"}
```

The response contains a grounded answer and the titles and filenames of the retrieved source documents. It does not include raw chunk text.

## Run the frontend

In another terminal:

```powershell
cd web
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open `http://localhost:3000`. Set `API_BASE_URL` in `web/.env.local` to the FastAPI origin (default `http://localhost:8000`). Next.js forwards browser `/api/*` requests to the backend.

## Structure

```text
app/api/       FastAPI, SQLAlchemy models, Alembic migrations, ingestion, embeddings, and retrieval
knowledge/     Source PDF documents
web/           Next.js App Router frontend, Tailwind CSS, and shadcn/ui components
```

The API and UI are kept in separate packages so future model-provider integrations can be added without coupling them to the landing page. Ingestion stores document text with an empty embedding; run the embedding command to populate vectors. There is no answer generation or authentication.
