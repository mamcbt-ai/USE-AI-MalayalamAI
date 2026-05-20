# USE-AI-MalayalamAI Backend

AI-powered Malayalam Voice-to-English Intelligence Platform Backend built using FastAPI, Whisper AI, and modern AI/NLP technologies.

---

# Project Overview

USE-AI-MalayalamAI is a next-generation multilingual AI voice platform focused on Malayalam (Kerala, India).

The backend provides:

* Malayalam speech recognition
* Real-time transcription
* Malayalam-to-English translation
* AI-powered text refinement
* Meeting summaries
* Voice assistant capabilities
* Payment and subscription APIs
* Authentication and user management

This backend is designed to support:

* Web applications
* Mobile apps
* SaaS platforms
* Enterprise integrations
* AI productivity systems

---

# Backend Architecture

```text
Frontend (Next.js / Mobile App)
            ↓
       WebSocket API
            ↓
      FastAPI Backend
            ↓
 ┌──────────────────────┐
 │  Audio Processing    │
 │  Whisper ASR         │
 │  Translation Engine  │
 │  AI Refinement       │
 │  Summarization       │
 └──────────────────────┘
            ↓
      Database Layer
(PostgreSQL / MongoDB / Redis)
            ↓
 Cloud Storage / AI Services
```

---

# Tech Stack

## Backend

* FastAPI
* Python
* WebSockets
* JWT Authentication
* Async APIs

## AI / NLP

* Whisper large-v3
* WhisperX
* IndicTrans2
* PyTorch

## Database

* PostgreSQL
* MongoDB
* Redis

## DevOps

* Docker
* GitHub
* Railway / Render / AWS

---

# Folder Structure

```text
backend/
│
├── app/
│   ├── routers/
│   │   ├── payment.py
│   │   ├── auth.py
│   │   ├── transcription.py
│   │   └── websocket.py
│   │
│   ├── services/
│   │   ├── asr_service.py
│   │   ├── translation_service.py
│   │   ├── summary_service.py
│   │   └── payment_service.py
│   │
│   ├── models/
│   ├── utils/
│   ├── core/
│   └── config/
│
├── main.py
├── requirements.txt
├── .env
├── Dockerfile
└── README.md
```

---

# Installation & Setup

## 1. Clone Repository

```bash
git clone https://github.com/mamcbt-ai/USE-AI-MalayalamAI.git
```

## 2. Navigate to Backend

```bash
cd USE-AI-MalayalamAI/backend
```

## 3. Create Virtual Environment

### Windows

```bash
python -m venv ml_env
```

Activate:

```bash
ml_env\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv ml_env
source ml_env/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install FFmpeg

Whisper requires FFmpeg.

### Windows

Download:
https://ffmpeg.org/download.html

Add FFmpeg to system PATH.

Verify installation:

```bash
ffmpeg -version
```

---

# Run Backend Server

```bash
uvicorn main:app --reload
```

Server runs at:

```text
http://127.0.0.1:8000
```

---

# API Documentation

FastAPI automatically generates API docs.

## Swagger UI

```text
http://127.0.0.1:8000/docs
```

## ReDoc

```text
http://127.0.0.1:8000/redoc
```

---

# API Endpoints

## Health Check

```http
GET /
```

Response:

```json
{
  "message": "Backend running successfully"
}
```

---

# Transcription API

```http
POST /transcribe
```

Purpose:

* Malayalam speech-to-text
* Whisper AI transcription

---

# Translation API

```http
POST /translate
```

Purpose:

* Malayalam → English translation
* Transliteration support

---

# Payment API

```http
POST /payment/create-order
```

Purpose:

* Razorpay payment integration
* Subscription handling

---

# Planned APIs

```http
POST /summary
POST /voice-command
POST /meeting-notes
POST /auth/login
POST /auth/register
POST /websocket/live-transcribe
```

---

# WebSocket Streaming Pipeline

Planned architecture:

```text
Browser Microphone
        ↓
WebSocket
        ↓
FastAPI
        ↓
Whisper AI
        ↓
Live Malayalam Transcript
        ↓
Translation Layer
        ↓
Frontend UI
```

---

# Environment Variables

Create `.env`

```env
OPENAI_API_KEY=your_key
RAZORPAY_KEY_ID=your_key
RAZORPAY_SECRET=your_secret
DATABASE_URL=your_database_url
SECRET_KEY=your_secret_key
```

---

# Docker Support

Build container:

```bash
docker build -t malayalam-ai-backend .
```

Run:

```bash
docker run -p 8000:8000 malayalam-ai-backend
```

---

# Security Features

Planned:

* JWT authentication
* HTTPS
* API rate limiting
* Role-based access
* Prompt injection protection
* Secure audio handling

---

# Development Roadmap

## Phase 1 — MVP

* [x] Backend setup
* [x] Whisper integration
* [x] Payment router
* [ ] WebSocket streaming
* [ ] Live transcription

## Phase 2 — AI Features

* [ ] Translation pipeline
* [ ] AI summaries
* [ ] Meeting assistant
* [ ] Email generation

## Phase 3 — SaaS Platform

* [ ] Authentication
* [ ] Subscription system
* [ ] Usage analytics
* [ ] Team collaboration

## Phase 4 — Scale

* [ ] Mobile app
* [ ] Offline mode
* [ ] GPU optimization
* [ ] Multi-language expansion

---

# Vision

The goal of USE-AI-MalayalamAI is to become a complete Malayalam AI Voice Intelligence Platform for:

* education
* accessibility
* productivity
* businesses
* enterprise communication
* AI-powered documentation

Focused initially on Kerala users and later expanding to multilingual Indian AI solutions.

---

# Author

Built by:
Mamcbt-AI

GitHub:
https://github.com/mamcbt-ai
