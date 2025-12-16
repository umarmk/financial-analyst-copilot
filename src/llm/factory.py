from __future__ import annotations

import os

from .ollama_client import OllamaClient
from .client import LLMClient

# Function to create LLM client 
# TODO: Add support for other providers 
def get_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        model = os.getenv("OLLAMA_MODEL", "phi3.5:3.8b-mini-instruct-q4_K_M").strip()
        return OllamaClient(base_url=base_url, model=model)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")