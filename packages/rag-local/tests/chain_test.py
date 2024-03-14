import pytest

from rag_local.loaders import (
    WebBaseLoaderComponent,
    JSONLoaderComponent,
    TextLoaderComponent,
)
from langchain_community.document_loaders import WebBaseLoader, TextLoader, JSONLoader

from rag_local.chain import build_loader


def test_build_loader():
    """Тест использует ссылки на файлы и на url для загрузки данных"""
    config = {
        "provider": "WebBaseLoader",
        "uri": "https://ummah.su/info/terminy-islama",
    }
    loader = build_loader(config=config)
    assert isinstance(loader, WebBaseLoader)

    config = {
        "provider": "JsonLoader",
        "uri": "data/quran_dict.json",
        "jq_schema": ".data[].tafsir_ru",
    }
    loader = build_loader(config=config)
    assert isinstance(loader, JSONLoader)

    config = {
        "provider": "TextLoader",
        "uri": "data/ru.kuliev.txt",
    }
    loader = build_loader(config=config)
    assert isinstance(loader, TextLoader)


# uri: data/quran.txt
# encoding: utf-8
