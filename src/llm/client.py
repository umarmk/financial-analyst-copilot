from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# Request object for the LLM
@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    system: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None   # Ollama may ignore for some models

# Return the model's response
class LLMClient(Protocol):
    def generate(self, request: LLMRequest) -> str: 
        ...



    
