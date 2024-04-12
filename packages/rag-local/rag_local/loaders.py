# -*- coding: utf-8 -*-
# loaders.py
"""
Module contains adapters classes for text loaders by different providers.
"""

# pylint: disable=no-name-in-module
import json
from typing import Dict, Sequence, Union
from bs4 import BeautifulSoup as Soup
from logger import logger
from langchain_community.document_loaders import WebBaseLoader, TextLoader, JSONLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from rag_local.component import raise_not_implemented

# from langchain_community.document_loaders.json_loader import JSONLoader

# from langchain_community.document_loaders import json_loader


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
        "quranjsonloader": QuranJSONLoaderComponent().build,
        "textloader": TextLoaderComponent().build,
        "recursiveloader": RecursiveUrlLoaderComponent().build,
    }
    loader = providers.get(config["provider"].lower(), raise_not_implemented)(**config)
    return loader


class WebBaseLoaderComponent:
    display_name = "WebBaseLoaderComponent"
    description = "Web Loader Component"
    documentation = ""

    def build(self, uri: Union[str, Sequence[str]] = "", **kwargs) -> WebBaseLoader:
        return WebBaseLoader(uri)


class RecursiveUrlLoaderComponent:
    display_name = "RecursiveUrlLoaderComponent"
    description = "Recursive URL Loader Component"
    documentation = ""

    def build(self, uri: str, max_depth: int = 2, **kwargs) -> RecursiveUrlLoader:
        return RecursiveUrlLoader(
            url=uri, max_depth=max_depth, extractor=lambda x: Soup(x, "html.parser")  # type: ignore
        )


class TextLoaderComponent:
    display_name = "TextLoaderComponent"
    description = "Text file Loader"
    documentation = ""

    def build(
        self, uri: Union[str, Sequence[str]] = "", encoding: str = "utf-8", **kwargs
    ) -> TextLoader:
        return TextLoader(file_path=uri, encoding=encoding)


class JSONLoaderComponent:
    display_name = "JsonLoaderComponent"
    description = "Json file Loader"
    documentation = ""

    def build(
        self,
        uri: Union[str, Sequence[str]] = ".",
        jq_schema: str = "",
        content_key: str = "",
        **kwargs,
    ) -> JSONLoader:

        return JSONLoader(
            file_path=uri,
            jq_schema=jq_schema,
            content_key=content_key,
            is_content_key_jq_parsable=True,
        )


class QuranJSONLoaderComponent:
    """
    Loader for structutred Quran JSON file.
    Json format:
    { data": [
        {
        "id": 1,  # № абсолютный номер аята
        "s_n": 1, # № суры
        "a_n": 1, # № аята в суре
        "text": "In the name of Allah, most benevolent, ever-merciful.",  # текст аята
        "tafsir": "Bismillah بِسْمِ اللَّـهِ is a verse of the Holy Qur'an...", # текст тафсира
        "source": "Quran.com",            # описание источника тафсира
        "name": "Maarif-ul-Quran",        # название источника тафсира
        "language": "English",            # язык
        "author": "Mufti Muhammad Shafi", # автор тафсира
        "translator": ""                  # переводчик
        },
        ...
    ]}
    """

    display_name = "JsonLoaderComponent"
    description = "Json file Loader"
    documentation = ""

    def build(
        self,
        uri: Union[str, Sequence[str]] = ".",
        **kwargs,
    ) -> JSONLoader:
        jq_schema = ".data[]"
        content_key = ".tafsir"
        return JSONLoader(
            file_path=uri,
            jq_schema=jq_schema,
            content_key=content_key,
            is_content_key_jq_parsable=True,
            metadata_func=self.metadata_func,
        )

    def metadata_func(self, record: dict, metadata: dict) -> dict:
        """Extract and transform metadata from record of json file."""
        ayah_id = record["id"]
        surah_n = record["s_n"]
        ayah_n = record["a_n"]
        source = record["source"]
        name = record["name"]
        author = record["author"]
        info = {
            "file": metadata.get("source", ""),
            "ayah_id": ayah_id,
            "surah_n": surah_n,
            "ayah_n": ayah_n,
            "source": source,
            "name": name,
            "author": author,
            "cite": f"Qu'ran ({surah_n}:{ayah_n})",
        }
        return info
