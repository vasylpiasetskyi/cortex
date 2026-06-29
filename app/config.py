from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql+asyncpg://cortex:cortex@localhost:5432/cortex"
    model: str = "gpt-4o-mini"
    embedding_backend: str = "openai"
    local_embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    system_prompt: str = "You are a helpful assistant."
    qdrant_url: str = "http://localhost:6333"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
