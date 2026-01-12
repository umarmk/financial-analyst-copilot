from __future__ import annotations

import os
from dotenv import load_dotenv

from .ollama_client import OllamaClient
from .client import LLMClient
from .openrouter_client import OpenRouterClient

# Factory function to create LLM client 
def get_llm_client() -> LLMClient:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        model = os.getenv("OLLAMA_MODEL", "phi3.5:3.8b-mini-instruct-q4_K_M").strip()
        return OllamaClient(base_url=base_url, model=model)
    
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing in .env")

        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free").strip()  
        timeout = int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60"))

        app_url = os.getenv("OPENROUTER_APP_URL", "").strip() or None
        app_title = os.getenv("OPENROUTER_APP_TITLE", "").strip() or None

        return OpenRouterClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout,
            app_url=app_url,
            app_title=app_title,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

# Helper function to get LLM provider
def get_llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower()