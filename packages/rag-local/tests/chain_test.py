import pytest

from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader, TextLoader, JSONLoader
from langchain_community.llms import VLLM, CTransformers, LlamaCpp
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings

from rag_local.chain import (
    build_embedder,
    build_loader,
    build_model,
    build_retriever,
    raise_not_implemented,
)
from rag_local.embeddings import GPT4AllEmbeddingsComponent


def test_build_loader():
    """Тест использует ссылки на файлы и на url для загрузки данных"""
    config = {
        "provider": "webbaseloader",
        "uri": "https://ummah.su/info/terminy-islama",
    }
    loader = build_loader(config=config)
    assert isinstance(loader, WebBaseLoader)

    config = {
        "provider": "jsonloader",
        "uri": "data/quran_dict.json",
        "jq_schema": ".data[].tafsir_ru",
    }
    loader = build_loader(config=config)
    assert isinstance(loader, JSONLoader)

    config = {
        "provider": "textloader",
        "uri": "data/ru.kuliev.txt",
    }
    loader = build_loader(config=config)
    assert isinstance(loader, TextLoader)

    # Проверяем, что выбрасывается исключение, если нет соответствующего провайдера
    config = {
        "provider": "SomeNotImplementedProvider",
        "uri": "any_uri",
    }
    try:
        # Должно быть возвращена функция raise_not_implemented
        loader = build_loader(config=config)
    except Exception as e:
        assert isinstance(e, NotImplementedError)


def test_build_model():

    config = {
        "provider": "llamacpp",
        "model": "models/saiga-mistral-7b",
        "model_file": "saiga-mistral-q4_K.gguf",
        "temperature": 0.1,
        "max_tokens": 4096,
        "top_k": 1,
        "top_p": 1.0,
        "n_ctx": 8192,
    }
    llm = build_model(config=config)
    assert isinstance(llm, LlamaCpp)

    config = {
        "provider": "ctransformers",
        "model": "IlyaGusev/saiga_mistral_7b_gguf",
        "model_file": "saiga-mistral-q4_K.gguf",
        "temperature": 0.1,
        "max_tokens": 4096,
        "top_k": 1,
        "top_p": 1.0,
        "n_ctx": 8192,
        "hf": True,
    }
    llm = build_model(config=config)
    assert isinstance(llm, CTransformers)

    # Проверяем, что выбрасывается исключение, если нет соответствующего провайдера
    config = {"provider": "SomeNotImplementedProvider"}
    try:
        llm = build_model(config=config)
    except Exception as e:
        assert isinstance(e, NotImplementedError)

    # TODO: add test for vllm


def test_build_embedder():
    # provider: GPT4AllEmbeddings
    # model_name:
    config = {
        "provider": "gpt4allembeddings",
        "model_name": "",
    }
    embedder = build_embedder(config=config)
    assert isinstance(embedder, GPT4AllEmbeddings)

    config = {
        "provider": "huggingfaceembeddings",
        "model_name": "sentence-transformers/all-mpnet-base-v2",
    }
    embedder = build_embedder(config=config)
    assert isinstance(embedder, HuggingFaceEmbeddings)

    config = {"provider": "SomeNotImplementedProvider"}
    try:
        # Должно быть возвращена функция raise_not_implemented
        embedder = build_embedder(config=config)
    except Exception as e:
        assert isinstance(e, NotImplementedError)


def test_build_retriever():
    config = {
        "provider": "chroma",
        "search_kwargs": {"k": 3},
        "anonymized_telemetry": False,
        "path": "./db",
    }
    texts = [
        "First document",
        "Second document",
        "Third document",
        "Fourth document",
    ]
    documents = [Document(page_content=text) for text in texts]
    embedder = build_embedder({"provider": "gpt4allembeddings", "model_name": ""})
    retreiver = build_retriever(documents=documents, embedder=embedder, config=config)
    assert retreiver

    config = {
        "provider": "SomeNotImplementedProvider",
    }
    try:
        # Должно быть возвращена функция raise_not_implemented
        retreiver = build_retriever(
            documents=documents, embedder=embedder, config=config
        )
    except Exception as e:
        assert isinstance(e, NotImplementedError)


#   path: ./db
#   search_kwargs:
#     k: 3
#   anonymized_telemetry: false
