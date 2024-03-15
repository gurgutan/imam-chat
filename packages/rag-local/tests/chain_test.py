import pytest

# from rag_local.loaders import (
#     WebBaseLoaderComponent,
#     JSONLoaderComponent,
#     TextLoaderComponent,
# )

# from rag_local.llms import VLLMComponent, CTransformersComponent, LlamaCppComponent

from langchain_community.document_loaders import WebBaseLoader, TextLoader, JSONLoader
from langchain_community.llms import VLLM, CTransformers, LlamaCpp

from rag_local.chain import build_loader, build_model, raise_not_implemented


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

    # Должно быть возвращена функция raise_not_implemented
    config = {
        "provider": "AnyOtherProvider",
        "uri": "any_uri",
    }
    # Проверяем, что выбрасывается исключение, если нет соответствующего провайдера
    try:
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

    # TODO: test for vllm
