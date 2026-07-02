from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llama_base_url: str = "http://localhost:8080"
    proxy_port: int = 8081
    erase_slot_id: int | None = None  # LLAMA_ERASE_SLOT_ID
    erase_timeout_s: float = 5.0
    upstream_timeout_s: float = 300.0
    detect_mode: str = "auto"  # HERMES_NEW_DETECT_MODE: auto | manual
    erase_enabled: bool = True
    max_observed_messages: int = 20


settings = Settings()
