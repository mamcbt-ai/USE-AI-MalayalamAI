from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Malayalam AI Agent"
    API_VERSION: str = "v1"
    CLAUDE_API_KEY: str = ""
    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""

    class Config:
        env_file = ".env"


settings = Settings()