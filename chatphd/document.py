import os
from .fcn import FCN_CONTENT
from .lns import LNS_CONTENT

def get_current_document_name() -> str:
    return os.getenv("DOCUMENT_NAME", "fcn")

def set_document(name: str): 
    os.environ["DOCUMENT_NAME"] = name

def get_document_content() -> str:
    if get_current_document_name() == "fcn":
        return FCN_CONTENT
    elif get_current_document_name() == "lns":
        return LNS_CONTENT
    else:
        raise ValueError(f"Document {get_current_document_name()} not found")


doc_name_map = {
    "fcn": "Combining supervised learning and local search for the multicommodity capacitated fixed-charge network design problem",
    "lns": "Supervised Large Neighbourhood Search for MIPs"
}

def get_document_full_name(short_name: str) -> str:
    return doc_name_map[short_name]

def get_document_short_name(full_name: str) -> str:
    return list(doc_name_map.keys())[list(doc_name_map.values()).index(full_name)]