from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .client import LLMClient, LLMRequest

# OpenRouter client implementation
@dataclass
class OpenRouterClient(LLMClient):
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 60
    app_url: str | None = None
    app_title: str | None = None

    # Generate a response
    def generate(self, request: LLMRequest) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions" # OpenRouter API endpoint

        # Prepare the request payload
        system_text = (request.system or "").strip()
        user_text = request.prompt.strip()

        # Build the messages array
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

        # Build the payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(request.temperature),
        }

        # Add max_tokens if provided
        if request.max_tokens is not None:
            payload["max_tokens"] = int(request.max_tokens)

        # Build the headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Add attribution headers if provided
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            headers["X-Title"] = self.app_title

        # Send the request
        data = json.dumps(payload).encode("utf-8")
        data = json.dumps(payload).encode("utf-8")
        http_request = Request(url, data=data, headers=headers)

        # Send the request and parse the response
        try:
            with urlopen(http_request, timeout=self.timeout_seconds) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            print(f"DEBUG: OpenRouter HTTP Error Body: {body}")
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}") from None
        except URLError as e:
            raise RuntimeError(f"OpenRouter connection error: {e}") from None

        # OpenAI-style response shape
        try:
            return (resp_json["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            # If schema differs / error response
            print(f"DEBUG: Parsing Error: {e}")
            raise RuntimeError(f"Unexpected OpenRouter response: {resp_json}") from None