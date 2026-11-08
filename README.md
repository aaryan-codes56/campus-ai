# Campus AI Assistant

Ask questions about university notices, policies and calendars, and get answers grounded in the source documents with clickable citations.

Work in progress. The full README arrives in Milestone 10.

## Local setup

Requires Docker (Colima, OrbStack or Docker Desktop), [uv](https://docs.astral.sh/uv/) and Node.js 26.

```bash
cp .env.example .env
docker compose up -d --wait db

# Backend, on http://localhost:8000
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload

# Frontend, on http://localhost:3000 (in a second terminal)
cd frontend
npm install
npm run dev
```

To run everything in containers instead: `docker compose --profile full up --build`.

## Ingesting documents

Put files in `data/raw/` (PDF, DOCX, HTML or TXT), describe them in `data/manifest.yaml`, set `VOYAGE_API_KEY` in `.env`, then:

```bash
cd backend
uv run python -m app.ingestion.cli            # ingest new or changed files, then embed
uv run python -m app.ingestion.cli --dry-run  # parse and chunk only, save nothing
uv run python -m app.ingestion.cli --no-embed # store chunks without calling Voyage
uv run python -m app.ingestion.cli --force    # re-process unchanged files
uv run python -m app.ingestion.cli --prune    # remove documents no longer on disk
```

Re-running is safe: unchanged files are skipped, and an interrupted embedding run resumes where it stopped.

## Searching from the terminal

```bash
cd backend
uv run python -m app.retrieval.cli "When are the mid term exams?"
uv run python -m app.retrieval.cli "When are the mid term exams?" --compare
uv run python -m app.retrieval.cli "Leave rules" --mode keyword --doc-type policy
```
