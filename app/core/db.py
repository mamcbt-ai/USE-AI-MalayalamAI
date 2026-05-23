import os
from sqlmodel import SQLModel, create_engine
from app.models.test_model import TestItem
from app.models.audio_record import AudioRecord
from app.models.user import User

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:admin123@localhost:5432/postgres")

# Fix for Railway: postgres:// -> postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    SQLModel.metadata.create_all(engine)
