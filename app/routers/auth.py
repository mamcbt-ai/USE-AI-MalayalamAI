from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlmodel import Session
from pydantic import BaseModel

from app.services.auth_service import (
    create_user, get_user_by_email,
    verify_password, create_access_token, decode_token
)
from app.core.db import engine

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ========================
# REQUEST MODELS
# ========================
class RegisterRequest(BaseModel):
    email: str
    password: str


# ========================
# REGISTER
# ========================
@router.post("/register")
def register(data: RegisterRequest):
    with Session(engine) as session:
        existing = get_user_by_email(session, data.email)
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        user = create_user(session, data.email, data.password)
        token = create_access_token({"sub": user.email})
        return {
            "message": "Account created successfully",
            "token": token,
            "email": user.email,
            "plan": user.plan
        }


# ========================
# LOGIN
# ========================
@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as session:
        user = get_user_by_email(session, form_data.username)
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        token = create_access_token({"sub": user.email})
        return {
            "access_token": token,
            "token_type": "bearer",
            "email": user.email,
            "plan": user.plan
        }


# ========================
# GET CURRENT USER
# ========================
@router.get("/me")
def get_me(token: str = Depends(oauth2_scheme)):
    email = decode_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")
    with Session(engine) as session:
        user = get_user_by_email(session, email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "email": user.email,
            "plan": user.plan,
            "is_active": user.is_active
        }