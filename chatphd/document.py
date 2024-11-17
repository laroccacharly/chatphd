from typing import Union
from .fcn import FCN_CONTENT
from .lns import LNS_CONTENT
from .lap import LAP_CONTENT
from pydantic import BaseModel

class Document(BaseModel):
    name: str
    full_name: str
    content: Union[str, None] = None

    def get_content(self) -> str:
        if self.content is None:
            self.content = self.load_content()
        return self.content
    
    def load_content(self):
        if self.name == "fcn":
            return FCN_CONTENT
        elif self.name == "lns":
            return LNS_CONTENT
        elif self.name == "lap":
            return LAP_CONTENT
        else:
            raise ValueError(f"Document {self.name} not found")


def get_all_document_names() -> list[str]:
    return ["lap", "fcn", "lns"]

def get_all_documents() -> list[Document]:
    return [Document(
        name=name,
        full_name=get_document_full_name(name),
    ) for name in get_all_document_names()]

doc_name_map = {
    'lap': 'One-shot Learning for MIPs with SOS1 Constraints', 
    "fcn": "Combining supervised learning and local search for the multicommodity capacitated fixed-charge network design problem",
    "lns": "Supervised Large Neighbourhood Search for MIPs",
}

def get_document_full_name(short_name: str) -> str:
    return doc_name_map[short_name]
