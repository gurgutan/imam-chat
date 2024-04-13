# import os
# from pprint import pprint

from typing import Dict
from logger import logger


# from langchain_core.prompts import ChatPromptTemplate
# pylint: disable=no-name-in-module
# pylint: disable=unused-import
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.llms import build_model
from rag.embeddings import build_embedder
from rag.loaders import build_loader
from rag.retrievers import build_retriever
from rag.prompts import MuslimImamPrompt
from rag.qadb import connect_qa_db, insert, select


# Add typing for input
class Question(BaseModel):
    __root__: str


def build_chain(config: Dict):
    """
    Build rag chain from config parameters
    """

    # if not (conn := connect_qa_db()):
    #     logger.error("Could not connect to QA DB")
    #     raise IOError("Could not connect to QA DB")

    # Выбираем загрузчик данных
    loader = build_loader(config["loader"])
    logger.info("Using loader %s", loader.__class__.__name__)

    # Загружаем данные из источника
    logger.info(f"Loading data with settings: {config['loader']} ...")
    data = loader.load()

    # Применяем сплиттер
    logger.info("Documents splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Produced %i chunks", len(chunks))

    # Выбираем эмбеддер
    embedder = build_embedder(config["embedder"])
    logger.info("Using embedder %s", embedder.__class__.__name__)

    # Добавляем vectorstore retriever
    logger.info("Creating data indexes with embedder")
    retriever = build_retriever(chunks, embedder, config["vectordb"])
    logger.info("Using retriever %s", retriever.__class__.__name__)

    # Prompt
    # prompt = ChatPromptTemplate.from_template(template)
    # prompt = QuestionAnswerCoTPrompt().build()
    prompt = MuslimImamPrompt().build()

    # Готовим модель
    logger.info(f"Initiating LLM: {config['llm']}")
    model = build_model(config["llm"])

    # Создаем RAG chain
    rag_chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})  # type: ignore
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain.with_types(input_type=Question)
