from .pdf import load_pdf_as_base64
from .claude_client import get_client
from .document import Document

def estimate_cost(token_count: int) -> float:
    print(f"Estimated cost: ${token_count / 1000000}")

def get_token_count() -> int: 
    pdf_base64 = load_pdf_as_base64()

    response = get_client().beta.messages.count_tokens(
        betas=["token-counting-2024-11-01", "pdfs-2024-09-25"],
        model="claude-3-5-sonnet-20241022",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_base64
                    }
                },
                {
                    "type": "text",
                    "text": "Please summarize this document."
                }
            ]
        }]
    )
    return response.input_tokens

def get_token_count_for_document(document: Document) -> int:
    document_content = document.get_content()
    response = get_client().beta.messages.count_tokens(
        betas=["token-counting-2024-11-01"],
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": document_content}]
    )
    estimate_cost(response.input_tokens)
    return response.input_tokens