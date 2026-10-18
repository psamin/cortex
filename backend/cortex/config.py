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
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # InsForge (database, storage, model gateway)
    insforge_base_url: str | None = None
    insforge_api_key: str | None = None  # server-side admin/API key, never sent to the browser
    insforge_bucket: str = "cortex-assets"
    vlm_model: str = "anthropic/claude-sonnet-5.5"
    agent_model: str = "anthropic/claude-sonnet-5.5"

    # Self-hosted open model (our own agent): any OpenAI-compatible endpoint, e.g. vLLM serving
    # Qwen3-VL on Futurama behind an SSH tunnel (scripts/vlm_futurama.sh). When PLANNER_BASE_URL is
    # set it is used for robot planning, VLM diagnostics and the studio assistant.
    planner_base_url: str | None = None  # e.g. http://127.0.0.1:8001/v1
    planner_api_key: str | None = None  # vLLM --api-key, if set
    selfhosted_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    # Qwen3-VL grounds in relative 0-1000 coordinates (QwenLM/Qwen3-VL cookbooks/2d_grounding.ipynb)
    selfhosted_point_format: str = "norm1000"

    # The studio's agent: the user's own Claude Code, run headless against the Cortex MCP server.
    # agent_engine: auto (Claude Code if the `claude` CLI and the MCP server are up) | claude_code | model | rules
    agent_engine: str = "auto"
    claude_code_model: str | None = None  # e.g. "sonnet" for faster turns; default = the user's Claude Code default
    hf_token: str | None = None  # optional: private Hugging Face datasets
    # Only needed on self-hosted InsForge; InsForge Cloud provisions it via its Model Gateway.
    openrouter_api_key: str | None = None

    # Training on the Futurama Slurm cluster (submitted over SSH)
    futurama_ssh_host: str | None = None  # e.g. pi34@planetexpress-login.cc.gatech.edu
    futurama_repo_dir: str = "~/cortex"
    futurama_partition: str = "general"
    futurama_gres: str = "gpu:l40s:1"

    # Approval token required for paid jobs / hardware when called via MCP
    cortex_approval_secret: str | None = None
    # Shared secret between the hosted MCP server and this backend (when tunneled)
    cortex_mcp_key: str | None = None
    # Real SO-101 arm: set only on the machine wired to the robot (spec section 5)
    cortex_enable_hardware: bool = False

    @property
    def insforge_enabled(self) -> bool:
        return bool(self.insforge_base_url and self.insforge_api_key)


@lru_cache
def settings() -> Settings:
    return Settings()
