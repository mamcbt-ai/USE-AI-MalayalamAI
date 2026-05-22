from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session
from pydantic import BaseModel
import razorpay
import hmac
import hashlib

from app.services.auth_service import decode_token, get_user_by_email
from app.core.db import engine

# ========================
# YOUR RAZORPAY KEYS
# ========================
RAZORPAY_KEY_ID = "rzp_test_SrSwGABhgvgkD9"
RAZORPAY_KEY_SECRET = "SaoDdIZ94nsdX3HC4iz3m0Oo"

client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

router = APIRouter(prefix="/payment", tags=["payment"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)):
    email = decode_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Please login")
    with Session(engine) as session:
        user = get_user_by_email(session, email)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user


# ========================
# PLANS
# ========================
PLANS = {
    "basic": {"amount": 9900, "name": "Basic Plan"},
    "pro": {"amount": 24900, "name": "Pro Plan"},
    "premium": {"amount": 49900, "name": "Premium Plan"},
}


# ========================
# CREATE ORDER
# ========================
class CreateOrderRequest(BaseModel):
    plan: str


@router.post("/create-order")
def create_order(data: CreateOrderRequest, current_user=Depends(get_current_user)):
    if data.plan not in PLANS:
        raise HTTPException(status_code=400, detail="Invalid plan")
    plan = PLANS[data.plan]
    order = client.order.create({
        "amount": plan["amount"],
        "currency": "INR",
        "payment_capture": 1
    })
    return {
        "order_id": order["id"],
        "amount": plan["amount"],
        "currency": "INR",
        "plan": data.plan,
        "key_id": RAZORPAY_KEY_ID
    }


# ========================
# VERIFY PAYMENT
# ========================
class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    plan: str


@router.post("/verify")
def verify_payment(data: VerifyPaymentRequest, current_user=Depends(get_current_user)):
    body = f"{data.razorpay_order_id}|{data.razorpay_payment_id}"
    expected_signature = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        body.encode(),
        hashlib.sha256
    ).hexdigest()

    if expected_signature != data.razorpay_signature:
        raise HTTPException(status_code=400, detail="Payment verification failed")

    with Session(engine) as session:
        user = get_user_by_email(session, current_user.email)
        user.plan = "pro"
        session.add(user)
        session.commit()

    return {
        "message": "Payment successful! Your plan has been upgraded.",
        "plan": "pro"
    }


# ========================
# GET PLANS
# ========================
@router.get("/plans")
def get_plans():
    return {
        "plans": [
            {"id": "basic", "name": "Basic Plan", "price": 99, "features": ["50 recordings/day", "All languages"]},
            {"id": "pro", "name": "Pro Plan", "price": 249, "features": ["200 recordings/day", "Priority processing"]},
            {"id": "premium", "name": "Premium Plan", "price": 499, "features": ["Unlimited recordings", "API access"]},
        ]
    }