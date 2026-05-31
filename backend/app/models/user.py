from sqlmodel import SQLModel, Field, Column
from sqlalchemy import Integer
from typing import Optional
from datetime import datetime

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    is_active: bool = Field(default=True)
    plan: str = Field(default="free")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    plan_expires_at: Optional[datetime] = Field(default=None)
    daily_limit: int = Field(default=10, sa_column=Column(Integer, nullable=False, server_default="10"))
