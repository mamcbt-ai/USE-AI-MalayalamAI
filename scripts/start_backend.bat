@echo off

echo =========================
echo Starting PostgreSQL...
echo =========================
docker start postgres

echo =========================
echo Starting Redis...
echo =========================
docker start redis

echo =========================
echo Starting Backend...
echo =========================

cd /d "C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\backend"

call ..\env\ml_env\Scripts\activate

uvicorn main:app --reload

pause