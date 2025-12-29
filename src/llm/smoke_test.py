import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.llm.client import LLMRequest
from src.llm.factory import get_llm_client


def main() -> None:
    print("Initializing LLM client...")
    try:
        client = get_llm_client()
        print(f"Client type: {type(client).__name__}")
        print(f"Model: {client.model}")
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        return

    print("Generating response...")
    try:
        request = LLMRequest(
            prompt="Reply with exactly: OK",
            temperature=0.0,
            max_tokens=100,
        )

        answer = client.generate(request)
        print("LLM answer:", repr(answer))
    except Exception as e:
        print(f"Generation failed: {e}")


if __name__ == "__main__":
    main()