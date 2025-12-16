from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.request import Request, urlopen

from .client import LLMClient, LLMRequest

@dataclass
class OllamaClient(LLMClient):
    base_url: str
    model: str
    timeout_seconds: int = 60
    
    def generate(self, request: LLMRequest) -> str:
        url = f"{self.base_url}/api/generate"
        
        # Keep payload small and predictable
        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": float(request.temperature),
            },
        }
        
        # Add system prompt if provided
        if request.system:
            payload["prompt"] = f"{request.system.strip()}\n\n{request.prompt}"

        # Add max tokens if provided
        if request.max_tokens is not None:
            payload["options"]["num_predict"] = int(request.max_tokens)

        data = json.dumps(payload).encode("utf-8")
        http_request = Request(url, data=data, headers={"Content-Type": "application/json"})

        # Make request
        with urlopen(http_request, timeout=self.timeout_seconds) as resp:
            resp_json = json.loads(resp.read().decode("utf-8"))

        return (resp_json.get("response") or "").strip()

