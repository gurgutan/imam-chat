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

from rag_local.llms import build_model
from rag_local.embeddings import build_embedder
from rag_local.loaders import build_loader
from rag_local.retrievers import build_retriever
from rag_local.prompts import MuslimImamPrompt
from rag_local.qadb import connect_qa_db, insert, select


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


# TODO: 1. Добавить логирование диалогов
# TODO: Добавить подключение папки с моделями (docker run -it -p 8010:8010/tcp -v ./db:/code/db -v ./models:/code/models imam-chat:latest )
# TODO: Добавить контроль длины промпта
# TODO: Добавить модуль splitters.py и раздел splitter в config.yml
# TODO: Добавить в цепь ConversationBufferMemory
# TODO: Добавить в ответ данные по источникам (метаданные в коллекцию документов и их извелечение)
# TODO: Поискать модель для арабского языка либо зафайнюнить
