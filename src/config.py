from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    def provider_config(prefix: str = "LLM") -> ProviderConfig:
        provider = os.getenv(f"{prefix}_PROVIDER", os.getenv("LLM_PROVIDER", "openai"))
        provider = provider.strip().lower()
        default_models = {
            "openai": "gpt-4o-mini",
            "custom": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash",
            "anthropic": "claude-3-5-haiku-latest",
            "ollama": "llama3.2",
            "openrouter": "openai/gpt-4o-mini",
        }
        key_names = {
            "openai": "OPENAI_API_KEY",
            "custom": "CUSTOM_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        base_names = {
            "custom": "CUSTOM_BASE_URL",
            "ollama": "OLLAMA_BASE_URL",
            "openrouter": "OPENROUTER_BASE_URL",
        }
        return ProviderConfig(
            provider=provider,
            model_name=os.getenv(f"{prefix}_MODEL", default_models.get(provider, "gpt-4o-mini")),
            temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
            api_key=os.getenv(key_names.get(provider, "")) or None,
            base_url=os.getenv(base_names.get(provider, "")) or None,
        )

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200")),
        compact_keep_messages=max(2, int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))),
        model=provider_config("LLM"),
        judge_model=provider_config("JUDGE"),
    )
