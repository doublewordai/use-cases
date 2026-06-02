"""Settings. One Doubleword key; everything else has a working default.

This project is batch-only: the whole point is to take a heavy eval workload and
run it on Doubleword's batch tier, so there are no realtime/async "mode" knobs.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Doubleword exposes the OpenAI-compatible batch endpoint with a *maximum*
# completion window. This is an SLA ceiling, not an expected wait — batches
# routinely finish in minutes. The long windows exist so the shared queue can
# guarantee a worst case across many tenants. Cheapest window = "24h".
CompletionWindow = Literal["24h", "1h"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    doubleword_api_key: str = Field(..., description="Single Doubleword key. No upstream provider keys.")
    doubleword_base_url: str = "https://api.doubleword.ai/v1"

    # Defaults to the same strong model for generation and judging — batch
    # pricing is what makes running a top-tier model as judge cheap.
    model_chat: str = "deepseek-ai/DeepSeek-V4-Pro"
    model_judge: str = "deepseek-ai/DeepSeek-V4-Pro"

    # Concurrency cap for the in-process hero script (eval.py) only.
    max_concurrency: int = 8

    # Arize Phoenix (local, via docker compose).
    project_name: str = "doubleword-arize"
    phoenix_collector_endpoint: str = "http://localhost:6006"

    # Maximum batch completion window. "24h" is cheapest; jobs usually finish
    # far sooner. "1h" is the express lane for smoke-testing.
    batch_completion_window: CompletionWindow = "24h"


settings = Settings()
