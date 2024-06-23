from pathlib import Path
from typing import Iterable

from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from ragsperiments.utils import EMBEDDING_FN, ROOT, get_chroma_langchain_client

_ = load_dotenv(find_dotenv())


def pdfs_to_chunks(paths: Iterable[Path]) -> list[Document]:
    raw_documents: list[Document] = []
    for f in paths:
        raw_documents += PyPDFLoader(f).load()

    semantic_splitter = SemanticChunker(EMBEDDING_FN)
    documents = semantic_splitter.transform_documents(raw_documents)

    return documents


if __name__ == '__main__':
    documents = pdfs_to_chunks(paths=(ROOT / 'data/raw').glob('*.pdf'))

    client = get_chroma_langchain_client()

    client.add_documents(documents)
