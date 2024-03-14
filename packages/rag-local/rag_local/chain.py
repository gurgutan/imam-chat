# Load
# import os
# from langchain_community.chat_models import ChatOllama
# from langchain_community.embeddings import GPT4AllEmbeddings
# from langchain_community.document_loaders import WebBaseLoader
from pprint import pprint
from typing import Dict
from logger import logger

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from rag_local.embeddings import (
    GPT4AllEmbeddingsComponent,
    HuggingFaceEmbeddingsComponent,
)

from rag_local.loaders import (
    WebBaseLoaderComponent,
    JSONLoaderComponent,
    TextLoaderComponent,
)

from rag_local.retrievers import ChromaRetreiverComponent

# Закомментирован baseline
# loader = WebBaseLoaderComponent().build("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()

# Split

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

# # Add to vectorDB
# vectorstore = Chroma.from_documents(
#     documents=all_splits,
#     collection_name="rag-private",
#     embedding=GPT4AllEmbeddingsComponent(),
# )
# retriever = vectorstore.as_retriever()

# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# LLM
# Select the LLM that you downloaded
# ollama_llm = "llama2:7b-chat"
# model = ChatOllama(model=ollama_llm)

# RAG chain
# chain = (
#     RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
#     | prompt
#     | model
#     | StrOutputParser()
# )


# Add typing for input
class Question(BaseModel):
    __root__: str


# chain = chain.with_types(input_type=Question)


def build_loader(config: Dict):
    """Build the loader based on the config dict

    Args:
        config (Dict): dictionary with the following keys:
            provider (str): WebBaseLoader | JsonLoader | TextLoader
            uri (str) : uri of the document to load
            jq_schema (str) : jq schema [https://python.langchain.com/docs/modules/data_connection/document_loaders/json]
            encoding (str): utf-8 | ascii

    Returns:
        loader

    Raises:
        Exception: NotImplementedError if
    """

    providers = {
        "WebBaseLoader": WebBaseLoaderComponent().build,
        "JsonLoader": JSONLoaderComponent().build,
        "TextLoader": TextLoaderComponent().build,
    }
    loader = providers.get(config["provider"], "JsonLoader")(**config)

    loader_provider = config["provider"]
    # if (loader_provider == "WebBaseLoader"):
    #     loader = WebBaseLoaderComponent().build(loader_uri)
    # elif (loader_provider == "TextLoader"):
    #     loader = TextLoaderComponent().build(loader_uri)
    # elif (loader_provider == "JsonLoader"):
    #     jq_schema = config.get("json_schema", ".data[].tafsir_ru")
    #     loader = JSONLoaderComponent().build(loader_uri, jq_schema)
    # else:
    #     raise NotImplementedError(f"Not implemented loader config: {config}")
    return loader


def build_chain(config: Dict):
    """
    Build rag chain from config parameters
    """
    # Выбираем загрузчик данных
    loader = build_loader(config["loader"])

    logger.info(f"Всего получено {len(chunks)} документов")

    # Загружаем данные из источника
    data = loader.load()

    # Применяем сплиттер
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)

    logger.info(f"Всего получено {len(chunks)} документов")

    # Выбираем эмбеддер
    embedder_config = config.get("embedder", "GPT4AllEmbeddings")
    embedder_provider = embedder_config["provider"]
    if embedder_provider == "GPT4AllEmbeddings":
        embedder = GPT4AllEmbeddingsComponent().build()
    elif embedder_provider == "HuggingFaceEmbeddings":
        embedder = HuggingFaceEmbeddingsComponent().build(embedder_config["model_name"])
    else:
        raise Exception(f"Not implemented embedder {embedder_config}")

    # Добавляем vectorstore retriever
    vectordb_config = config.get("vectordb")
    vectordb_provider = vectordb_config.get("provider", "chroma")
    if vectordb_provider == "chroma":
        retriever = ChromaRetreiverComponent().build(
            documents=chunks,
            embedder=embedder,
            search_kwargs={"k": vectordb_config.get("k", 1)},
        )
    else:
        raise Exception(f"Not implemented retriever {vectordb_config}")
    # if (vectordb_provider == "chroma"):
    #     retriever = Chroma.from_documents(
    #         documents=chunks,
    #         collection_name="rag-local",
    #         embedding=embedder
    #     ).as_retriever(search_kwargs={"k": vectordb_config.get("k", 1)})

    # Prompt
    template = config.get(
        "prompt_template",
        "Answer the question based only on the following context:\n{context}\nQuestion: {question}",
    )
    prompt = ChatPromptTemplate.from_template(template)

    # Callbacks для поддержки стриминга
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Готовим локальную модель
    llm_config = config["llm"]
    model_name = llm_config.get("model_name", "models/saiga-mistral-q4_K.gguf")
    temperature = llm_config.get("temperature", 0.1)
    max_tokens = llm_config.get("max_tokens", 512)
    top_p = llm_config.get("top_p", 0.9)
    n_ctx = llm_config.get("n_ctx", 4096)
    model = LlamaCpp(
        model_path=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        callback_manager=callback_manager,
        n_ctx=n_ctx,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    # RAG chain
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.with_types(input_type=Question)


# TODO: Добавить контроль длины промпта
# TODO: Добавить тесты
# TODO: Сделать модуль ретриверов rag_local/retrievers.py
# TODO: Добавить в цепь ConversationBufferMemory
# TODO: Добавить в ответ данные по источникам (метаданные)
# TODO: Добавить llms.py
