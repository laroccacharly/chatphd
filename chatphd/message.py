from .claude_client import get_client
from .document import Document
from typing import Iterator
import os

def get_model_name() -> str:
    return os.getenv("MODEL_NAME", "claude-3-5-haiku-20241022")

def get_max_tokens() -> int:
    return int(os.getenv("MAX_TOKENS", 500))

def get_message_stream(messages: list[dict], document: Document) -> Iterator[str]:
    client = get_client()
    return client.messages.create(
        model=get_model_name(),
        system=document.load_content(),
        messages=messages,
        max_tokens=get_max_tokens(),
        stream=True
    )