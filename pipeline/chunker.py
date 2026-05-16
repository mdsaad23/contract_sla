from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from typing import List


def chunk_contract(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[str]:
    doc = Document(text=text)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([doc])
    return [node.get_content() for node in nodes]
