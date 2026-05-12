from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Malayalam AI Agent"
    API_VERSION: str = "v1"

    class Config:
        env_file = ".env"

settings = Settings()
