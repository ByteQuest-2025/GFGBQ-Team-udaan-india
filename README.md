🚀 Project

Problem Statement : Predictive Hospital Resource & Emergency Load Intelligence System

HELLIOS • UDAAN INDIA

Project Presentation: https://docs.google.com/presentation/d/1mUZi1sy3XA6uUBTHOKY5RsZs0VnvQYQW/edit?usp=drive_link&ouid=107715756340849135526&rtpof=true&sd=true

A full‑stack dashboard that runs a hospital analytics pipeline and presents ICU capacity risk, ED admissions forecasting, staffing signals, and explainability in a clean, demo‑ready UI.

Status: Active development • Made with React • License: MIT

📌 Table of Contents

- Overview
- Features
- Tech Stack
- Project Structure
- Installation
- Usage
- Screenshots
- Deployment
- Future Enhancements
- Contributing
- License
- Author

🔍 Overview

This project helps hospitals proactively plan for demand by combining a Python data/forecasting pipeline with a modern React dashboard.

It includes:

- A FastAPI backend that exposes dashboard APIs (`/api/ui/dashboard`, `/api/run`) and health/metrics endpoints.
- A Vite + React frontend that visualizes KPIs, forecasts, ICU utilization, system health, and recommended actions.
- A shared pipeline controller that runs end‑to‑end forecasting/alerts in one consistent execution path.

✨ Features

- ⚡ Fast, responsive dashboard UI
- 🧩 Component-based architecture with reusable UI primitives
- 📈 ED admissions forecast visualization (24h/48h/7d views)
- 🏥 ICU capacity risk + monitoring mode (suppresses overflow alerts when occupancy is 0)
- 🧠 Explainability-friendly payloads for demo clarity
- ✅ Action confirmations via toasts (non-blocking UX)
- 🛠 Health endpoints + metrics-ready backend

🛠 Tech Stack

| Technology | Usage |
|---|---|
| React + TypeScript | Frontend UI |
| Vite | Frontend dev server + build |
| Tailwind CSS | Styling |
| FastAPI | Backend API |
| Uvicorn | ASGI server |
| Python (pandas / numpy / scikit-learn) | Data processing + forecasting |
| Streamlit | Optional pipeline runner UI (local) |
| Docker + Docker Compose | Containerized run |

📂 Project Structure

project-root/
├── backend/
│   ├── app.py
│   ├── pipeline_service.py
│   └── ...
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── components/
│       ├── lib/
│       └── styles/
├── pipeline.py
├── alerts.py
├── dashboard_app.py
├── requirements.txt
├── docker-compose.yml
└── README.md

⚙️ Installation

Follow the steps below to run the project locally:

1) Clone the repository

```bash
git clone https://github.com/ByteQuest-2025/GFGBQ-Team-udaan-india.git
cd GFGBQ-Team-udaan-india
```

2) Backend (Python)

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

3) Frontend (Node)

```bash
cd frontend
npm install
```

▶️ Usage

Option A — Run locally (recommended for development)

1) Start the backend API (from repo root)

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8001
```

2) Start the frontend dev server (from `frontend/`)

```bash
npm run dev
```

Then open the Vite URL shown in the terminal (usually `http://localhost:5173`).

Option B — Run with Docker Compose

```bash
docker compose up --build
```

Backend endpoints (typical):

- Health: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`

🖼 Screenshots

Add screenshots or GIFs here to showcase your UI.

Example:

![Dashboard](screenshots/dashboard.png)

🚀 Deployment

You can deploy using:

- Vercel / Netlify (frontend)
- A VM/container platform (backend)

Build frontend:

```bash
cd frontend
npm run build
```

🔮 Future Enhancements

- 🔐 Authentication + role-based access
- 📊 Improved analytics + trend comparisons
- 🌙 Dark mode
- 📱 Mobile-first refinements
- 🧪 Automated tests + CI pipeline

🤝 Contributing

Contributions are welcome!

1) Fork the repository
2) Create a new branch (`feature/new-feature`)
3) Commit your changes
4) Push to the branch
5) Open a Pull Request

📄 License

This project is licensed under the MIT License.

👤 Author

Team Udaan India (ByteQuest 2025)

GitHub: https://github.com/ByteQuest-2025/GFGBQ-Team-udaan-india

⭐ If you like this project, don’t forget to star the repository!



