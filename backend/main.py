from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from app.core.config import settings
from app.core.db import init_db
from app.routers import home, audio, auth, payment

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title=settings.APP_NAME, version=settings.API_VERSION)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"message": "Too many requests. Please wait a moment and try again."},
    )


@app.on_event("startup")
def on_startup():
    init_db()
    print("DB init OK")


# ── CORS — allow all Vercel deployments + localhost ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://malayalam-ai-frontend.vercel.app",
        "https://malayalam-ai-frontend-hodophile.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_origin_regex=r"https://malayalam-ai-frontend.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(home.router)
app.include_router(audio.router)
app.include_router(auth.router)
app.include_router(payment.router)
