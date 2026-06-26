from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql+asyncpg://cortex:cortex@localhost:5432/cortex"
    model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
