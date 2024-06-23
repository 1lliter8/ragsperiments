from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from ragsperiments.questions import QUESTIONS
from ragsperiments.utils import LLM, get_chroma_langchain_client


def build_rag_chain() -> Runnable:
    client = get_chroma_langchain_client()

    retriever = client.as_retriever()

    prompt = (
        'You are an assistant for question-answering tasks. '
        'Use the following pieces of retrieved context to answer '
        "the question. If you don't know the answer, just say "
        "that you don't know. Use three sentences maximum and "
        'keep the answer concise. \n'
        'Question: {question} \n'
        'Context: {context} \n'
        'Answer:'
    )

    def format_docs(docs: Document):
        return '\n\n'.join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x['context'])))
        | ChatPromptTemplate.from_template(prompt)
        | LLM
        | StrOutputParser()
    )

    retrieve_docs = (lambda x: x['question']) | retriever

    return RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )


if __name__ == '__main__':
    chain = build_rag_chain()

    response = chain.invoke({'question': QUESTIONS[34]})

    print(response)
