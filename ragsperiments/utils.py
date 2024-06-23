from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
EMBEDDING_FN = OpenAIEmbeddings()
LLM = ChatOpenAI(model='gpt-4o')


def get_chroma_client_and_collection() -> tuple[ClientAPI, Collection]:
    client = chromadb.PersistentClient(path=str(DATA / 'chroma'))
    collection = client.get_or_create_collection(name='documents')
    return client, collection


def get_chroma_langchain_client() -> Chroma:
    client, collection = get_chroma_client_and_collection()
    return Chroma(
        client=client,
        collection_name=collection.name,
        embedding_function=EMBEDDING_FN,
    )
