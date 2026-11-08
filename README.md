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
