from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: Literal["openai", "anthropic"] = "openai"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    # API security settings
    # Comma-separated list of valid API keys. If empty, auth is disabled.
    api_keys: str = ""
    # Rate limit: max requests per window per API key
    rate_limit_rpm: int = 20       # requests per minute
    rate_limit_enabled: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def api_keys_list(self) -> list[str]:
        """Parse comma-separated API keys into a list."""
        if not self.api_keys:
            return []
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


settings = Settings()
