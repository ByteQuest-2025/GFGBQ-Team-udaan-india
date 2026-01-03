Predictive Hospital Resource & Emergency Load Intelligence System




HELLIOS




UDAAN INDIA


https://docs.google.com/presentation/d/1mUZi1sy3XA6uUBTHOKY5RsZs0VnvQYQW/edit?usp=drive_link&ouid=107715756340849135526&rtpof=true&sd=true



## Running locally (without Docker)

This repository contains:

- A FastAPI backend under `backend/`.
- A React/Vite frontend under `frontend/`.

To run locally using Python and Node:

1. Create and activate a virtual environment, then install Python deps:

	`pip install -r requirements.txt`

2. Start the backend API from the repo root:

	`uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload`

3. In a separate terminal, install frontend deps and start Vite dev:

	- `cd frontend`
	- `npm install`
	- `npm run dev`

## Running with Docker

You can run the full stack (backend API + frontend UI) with Docker Compose:

1. Build and start services from the repo root:

	`docker compose up --build`

2. Access the services:

	- Backend API: `http://localhost:8000` (docs at `/docs`, health at `/health` and `/health/ready`)
	- Frontend UI: `http://localhost:3000`

The backend stores monitoring history in a SQLite database mounted on the `backend-data` volume defined in `docker-compose.yml`.



