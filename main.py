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

from app.routers import home
from app.routers import audio
from app.routers import auth
from app.routers import payment

# =========================
# Load Environment Variables
# =========================
load_dotenv()

# =========================
# Rate Limiter
# =========================
limiter = Limiter(key_func=get_remote_address)

# =========================
# FastAPI App
# =========================
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.API_VERSION
)

# =========================
# Add Rate Limit Middleware
# =========================
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# =========================
# Handle 429 Errors
# =========================
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={
            "message": "Too many requests. Please wait a moment and try again."
        },
    )

# =========================
# Startup Event
# =========================
@app.on_event("startup")
def on_startup():
    print("Startup: skipping init_db for debug")

# =========================
# CORS Middleware
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Include Routers
# =========================
app.include_router(home.router)
app.include_router(audio.router)
app.include_router(auth.router)
app.include_router(payment.router)
