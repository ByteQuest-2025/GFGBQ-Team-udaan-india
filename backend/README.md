# Backend (FastAPI)

This backend exposes JSON endpoints backed by the existing hospital analytics pipeline in the repo.

## Run

From the repo root:

- Start the API server:

  `C:/Users/dasne/Desktop/udaanindia/.venv/Scripts/python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000`

- Open API docs:

  `http://127.0.0.1:8000/docs`

## Endpoints

- `GET /health`
- `POST /api/run` (run pipeline with params)
- `GET /api/dashboard` (cached last result or runs defaults)
- `GET /api/alert`
- `GET /api/admissions`
- `GET /api/icu`
- `GET /api/staff`

## CORS

CORS defaults to allowing Vite dev servers at `http://localhost:5173` and `http://127.0.0.1:5173`.
Override with:

- `CORS_ORIGINS=http://localhost:5173,http://localhost:3000`
