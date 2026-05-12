from sqlmodel import SQLModel, create_engine
from app.models.test_model import TestItem

DATABASE_URL = "postgresql://postgres:admin123@localhost:5432/postgres"

engine = create_engine(DATABASE_URL, echo=True)


def init_db():
    SQLModel.metadata.create_all(engine)
