from .claude_client import get_client
from .document import Document
from typing import Iterator
from .prompt import get_prompt
import os

def get_model_name() -> str:
    return os.getenv("MODEL_NAME", "claude-3-5-haiku-20241022")

def get_max_tokens() -> int:
    return int(os.getenv("MAX_TOKENS", 500))

def get_message_stream(messages: list[dict], document: Document) -> Iterator[str]:
    client = get_client()
    system_prompt = get_prompt(document.load_content())
    return client.messages.create(
        model=get_model_name(),
        system=system_prompt,
        messages=messages,
        max_tokens=get_max_tokens(),
        stream=True
    )