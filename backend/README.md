# Backend (FastAPI)

This backend exposes JSON endpoints backed by the existing hospital analytics pipeline in the repo.

## Run (development)

From the repo root with the project virtualenv activated:

- Start the API server (auto-reload, dev):

  `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload`

- Open API docs:

  `http://127.0.0.1:8000/docs`

## Run (production)

In production use a process manager and a production-ready ASGI server,
for example:

```bash
uvicorn backend.app:app \
  --host 0.0.0.0 --port 8000 \
  --workers 4
```

Recommended environment variables:

- `APP_ENV` – e.g. `development`, `staging`, `production`.
- `APP_DEBUG` – `true` / `false`.
- `CORS_ORIGINS` – comma-separated list of allowed frontend origins.

## Endpoints

- `GET /health`
- `GET /health/ready`
- `POST /api/run` (run pipeline with params)
- `GET /api/dashboard` (cached last result or runs defaults)
- `GET /api/monitoring/last-run` (latest monitoring metrics)
- `GET /api/monitoring/history?limit=N` (recent monitoring runs from SQLite)
- `GET /metrics` (Prometheus scrape endpoint)
- `GET /api/alert`
- `GET /api/admissions`
- `GET /api/icu`
- `GET /api/staff`

## CORS

CORS defaults to allowing Vite dev servers at `http://localhost:5173` and `http://127.0.0.1:5173`.
Override with:

- `CORS_ORIGINS=http://localhost:5173,http://localhost:3000`
