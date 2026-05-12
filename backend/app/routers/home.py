from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def home():
    return {"message": "Malayalam AI Backend is running!"}
