from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class AudioRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    language: str
    transcript: str
    translation: str
    malayalam_output: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
