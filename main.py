from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage

#!/usr/bin/env python
"""A more complex example that shows how to configure index name at run time."""
from typing import Any, Iterable, List, Optional, Type, Dict, Union

from fastapi import FastAPI
from langchain.schema.vectorstore import VST
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableFieldSingleOption,
    RunnableConfig,
    RunnableSerializable, Runnable,
)
from langchain_core.runnables import RunnableLambda

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser


from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

file_path = "./Paper.pdf"
loader = PyPDFLoader(file_path)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)



docs = loader.load()
print(f"Document loaded, logging the first few content: {docs[0].page_content[0:100]}")


llm = ChatOpenAI(model="gpt-4o")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
chain = rag_chain

def get_answer(query: str):
    return StrOutputParser().invoke(chain.invoke({"input":query})['answer'])


results = rag_chain.invoke({"input": "What is \"The Loop?\""})
print(results['answer'])

add_routes(app,RunnableLambda(get_answer))

# Add the routes with the custom runnable

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
