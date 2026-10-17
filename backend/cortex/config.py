"""Settings from environment / .env. Every provider is optional; missing
credentials select the disclosed local fallback."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(ROOT / ".env",), extra="ignore")

    data_dir: Path = ROOT / ".cortex-data"
    public_url: str = "http://localhost:8000"
    cors_origins: str = "http://localhost:5173"

    # InsForge (database, storage, model gateway)
    insforge_base_url: str | None = None
    insforge_api_key: str | None = None  # server-side admin/API key, never sent to the browser
    insforge_bucket: str = "cortex-assets"
    vlm_model: str = "anthropic/claude-sonnet-5.5"
    agent_model: str = "anthropic/claude-sonnet-5.5"
    # Only needed on self-hosted InsForge; InsForge Cloud provisions it via its Model Gateway.
    openrouter_api_key: str | None = None
    # Direct OpenAI route for GPT-6 Astra, used only if the InsForge gateway is unavailable.
    openai_api_key: str | None = None
    planner_model: str = "gpt-6-astra"

    # Training on the Futurama Slurm cluster (submitted over SSH)
    futurama_ssh_host: str | None = None  # e.g. pi34@planetexpress-login.cc.gatech.edu
    futurama_repo_dir: str = "~/cortex"
    futurama_partition: str = "general"
    futurama_gres: str = "gpu:l40s:1"

    # Approval token required for paid jobs / hardware when called via MCP
    cortex_approval_secret: str | None = None
    # Shared secret between the hosted MCP server and this backend (when tunneled)
    cortex_mcp_key: str | None = None

    @property
    def insforge_enabled(self) -> bool:
        return bool(self.insforge_base_url and self.insforge_api_key)


@lru_cache
def settings() -> Settings:
    return Settings()
