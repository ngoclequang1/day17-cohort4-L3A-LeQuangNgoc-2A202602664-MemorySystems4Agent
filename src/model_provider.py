from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Student TODO: map aliases like `anthorpic` -> `anthropic`."""

    normalized = (value or "openai").strip().lower().replace("-", "_")
    aliases = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google_genai": "gemini",
        "open_router": "openrouter",
        "openai_compatible": "custom",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if normalized not in supported:
        raise ValueError(f"Unsupported provider: {value!r}. Choose one of: {', '.join(sorted(supported))}")
    return normalized


def build_chat_model(config: ProviderConfig):
    """Student TODO: instantiate the real chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """

    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    try:
        if provider in {"openai", "custom"}:
            from langchain_openai import ChatOpenAI

            if config.api_key:
                common["api_key"] = config.api_key
            if provider == "custom":
                if not config.base_url:
                    raise ValueError("CUSTOM_BASE_URL is required for provider 'custom'.")
                common["base_url"] = config.base_url
            return ChatOpenAI(**common)
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            if config.api_key:
                common["google_api_key"] = config.api_key
            return ChatGoogleGenerativeAI(**common)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            if config.api_key:
                common["api_key"] = config.api_key
            return ChatAnthropic(**common)
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            if config.base_url:
                common["base_url"] = config.base_url
            return ChatOllama(**common)
        if provider == "openrouter":
            try:
                from langchain_openrouter import ChatOpenRouter

                if config.api_key:
                    common["api_key"] = config.api_key
                return ChatOpenRouter(**common)
            except ImportError:
                from langchain_openai import ChatOpenAI

                common["base_url"] = config.base_url or "https://openrouter.ai/api/v1"
                if config.api_key:
                    common["api_key"] = config.api_key
                return ChatOpenAI(**common)
    except ImportError as exc:
        raise RuntimeError(f"Missing LangChain integration package for provider '{provider}'.") from exc

    raise AssertionError("Provider normalization returned an unexpected value.")
