# import os
# from pprint import pprint
from typing import Dict
from logger import logger

# from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

# from langchain_core.prompts import ChatPromptTemplate
# pylint: disable=no-name-in-module
# pylint: disable=unused-import
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_local.llms import (
    LlamaCppComponent,
    CTransformersComponent,
    VLLMComponent,
    OpenAIComponent,
)

from rag_local.embeddings import (
    GPT4AllEmbeddingsComponent,
    HuggingFaceEmbeddingsComponent,
)

from rag_local.loaders import (
    RecursiveUrlLoaderComponent,
    WebBaseLoaderComponent,
    JSONLoaderComponent,
    TextLoaderComponent,
)

from rag_local.retrievers import ChromaRetreiverComponent

from rag_local.prompts import (
    QuestionAnswerPrompt,
    QuestionAnswerCoTPrompt,
)

from langchain_core.runnables.history import RunnableWithMessageHistory


# Add typing for input
class Question(BaseModel):
    __root__: str


def raise_not_implemented(**kwargs):
    """Raise NotImplementedError with kwargs as message."""
    raise NotImplementedError(f"Not implemented config: {kwargs}")


def build_loader(config: Dict):
    """Build the loader based on the config dict

    Args:
        config (Dict): dictionary with the following keys:
            provider (str): WebBaseLoader | JsonLoader | TextLoader | RecursiveUrlLoader
            uri (str) : uri of the document to load
            jq_schema (str) : jq schema [https://python.langchain.com/docs/modules/data_connection/document_loaders/json]
            encoding (str): utf-8 | ascii

    Returns:
        loader

    Raises:
        Exception: NotImplementedError if unknown provider
    """

    providers = {
        "webbaseloader": WebBaseLoaderComponent().build,
        "jsonloader": JSONLoaderComponent().build,
        "textloader": TextLoaderComponent().build,
        "recursiveloader": RecursiveUrlLoaderComponent().build,
    }
    loader = providers.get(config["provider"].lower(), raise_not_implemented)(**config)
    return loader


def build_model(config: Dict):
    """Build the model based on the config dict

    Args:
        config (Dict): dictionary with the following keys:
            provider (str): LlamaCpp | CTransformers | VLLM
            model_name (str) : path to the model or name of the model in huggingface hub

    Returns:
        model

    Raises:
        Exception: NotImplementedError if unknown provider
    """
    providers = {
        "llamacpp": LlamaCppComponent().build,
        "ctransformers": CTransformersComponent().build,
        "vllm": VLLMComponent().build,
        "openai": OpenAIComponent().build,
    }
    model = providers.get(config["provider"].lower(), raise_not_implemented)(**config)
    return model


def build_embedder(config: Dict):
    """Build the embedder based on the config dict

    Args:
        config (Dict): dictionary with the following keys:
            provider (str): GPT4AllEmbeddings | HuggingFaceEmbeddings
            model_name (str) : path to gpt4all model or huggingface embedder name
    """

    providers = {
        "gpt4allembeddings": GPT4AllEmbeddingsComponent().build,
        "huggingfaceembeddings": HuggingFaceEmbeddingsComponent().build,
    }
    embedder = providers.get(config["provider"].lower(), raise_not_implemented)(
        **config
    )
    return embedder


def build_retriever(documents, embedder, config: Dict):
    """Build the retriever based on the config dict"""
    providers = {
        "chroma": ChromaRetreiverComponent().build,
    }
    retriever = providers.get(config["provider"].lower(), raise_not_implemented)(
        documents=documents, embedder=embedder, **config
    )
    return retriever


def build_chain(config: Dict):
    """
    Build rag chain from config parameters
    """
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
    prompt = QuestionAnswerPrompt().build()

    # Готовим модель
    logger.info(f"Initiating LLM: {config['llm']}")
    model = build_model(config["llm"])
    # logger.info("Используется модель %s", model.__class__.__name__)

    # Создаем RAG chain
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})  # type: ignore
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.with_types(input_type=Question)


# TODO: Добавить контроль длины промпта
# TODO: Добавить модуль splitters.py и раздел splitter в config.yml
# TODO: Добавить в цепь ConversationBufferMemory
# TODO: Добавить в ответ данные по источникам (метаданные)
