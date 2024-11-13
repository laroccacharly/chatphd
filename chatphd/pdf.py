import base64
import os

def get_pdf_path() -> str:
    return os.getenv("PDF_PATH")

def load_pdf_as_base64(pdf_path: str = get_pdf_path()) -> str:
    with open(pdf_path, "rb") as pdf_file:
        return base64.standard_b64encode(pdf_file.read()).decode("utf-8")
