from __future__ import annotations

from src.llm.client import LLMRequest
from src.llm.factory import get_llm_client


def main() -> None:
    client = get_llm_client()

    request = LLMRequest(
        prompt="Reply with exactly: OK",
        temperature=0.0,
        max_tokens=10,
    )

    answer = client.generate(request)
    print("LLM answer:", repr(answer))


if __name__ == "__main__":
    main()