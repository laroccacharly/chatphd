import os
def get_document_content() -> str:
    document_path = os.getenv("DOCUMENT_PATH")
    with open(document_path, "r") as f:
        return f.read()

def set_document(name: str): 
    os.environ["DOCUMENT_PATH"] = f"documents/{name}.tex"


doc_name_map = {
    "fcn": "Combining supervised learning and local search for the multicommodity capacitated fixed-charge network design problem",
    "lns": "Supervised Large Neighbourhood Search for MIPs"
}

def get_document_full_name(short_name: str) -> str:
    return doc_name_map[short_name]

def get_document_short_name(full_name: str) -> str:
    return list(doc_name_map.keys())[list(doc_name_map.values()).index(full_name)]