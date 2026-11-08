FROM python:3.12-slim

ENV UV_NO_CACHE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
RUN pip install --no-cache-dir uv

WORKDIR /app
# Dependencies first, so code edits don't reinstall everything on rebuild
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
# Render supplies $PORT; locally it falls back to 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
