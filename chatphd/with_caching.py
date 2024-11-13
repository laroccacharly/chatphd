from .claude_client import client
from .pdf import load_pdf_as_base64

def chat_with_caching(message: str = "Summarize this document.") -> str:
    pdf_data = load_pdf_as_base64()
    response = client.beta.messages.create(
        model="claude-3-5-sonnet-20241022",
        betas=["pdfs-2024-09-25"], # "prompt-caching-2024-07-31"
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data
                        },
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": message
                    }
                ]
            }
        ],
    )
    return response.content[0].text