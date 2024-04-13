# -*- coding: utf-8 -*-
# loaders.py
"""
Module contains adapters classes for text loaders by different providers.
"""

# pylint: disable=no-name-in-module
import json
import os
from typing import Dict, Iterator, List, Sequence, Union
from bs4 import BeautifulSoup as Soup
from logger import logger
from langchain_community.document_loaders import WebBaseLoader, TextLoader, JSONLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.documents import Document

from rag.component import raise_not_implemented

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
    description = "Web Loader Component"
    documentation = ""

    def build(self, uri: Union[str, Sequence[str]] = "", **kwargs) -> WebBaseLoader:
        return WebBaseLoader(uri)


class RecursiveUrlLoaderComponent:
    description = "Recursive URL Loader Component"
    documentation = ""

    def build(self, uri: str, max_depth: int = 2, **kwargs) -> RecursiveUrlLoader:
        return RecursiveUrlLoader(
            url=uri, max_depth=max_depth, extractor=lambda x: Soup(x, "html.parser")  # type: ignore
        )


class TextLoaderComponent:
    description = "Text file Loader"
    documentation = ""

    def build(
        self, uri: Union[str, Sequence[str]] = "", encoding: str = "utf-8", **kwargs
    ) -> TextLoader:
        return TextLoader(file_path=uri, encoding=encoding)


class JSONLoaderComponent:
    description = "Json file Loader"
    documentation = ""

    def build(
        self,
        uri: str = ".",
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

    description = "Json file Loader"
    documentation = ""

    def build(
        self,
        uri: str = ".",
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

    def load(
        self,
        uri: Union[str, Sequence[str]] = ".",
        **kwargs,
    ) -> list[Document]:
        """
        Loads json files
        """
        if os.path.exists(uri):
            files = [os.path.isfile(filename) for filename in os.listdir(uri)]
        elif os.path.isfile(uri):
            files = [os.path.isfile(filename) for filename in [uri]]
        else:
            logger.error("File %s does not exist", uri)
        if not files:
            logger.error("No files found in %s", uri)
            return []

        docs = []
        for file in files:
            json_loader = self.build(file)
            docs.extend(json_loader.load())
            # Move file to 'finished' folder
            new_filename = os.path.dirname(file) + "/finished/" + os.path.basename(file)
            logger.info(f"Moving {file} to {new_filename}")
            os.renames(file, new_filename)

        return docs


def move_to_finished_folder(files: list[str]):
    """Moves files to dir ./finished/ near the file"""

    for file in filter(lambda x: x, files):
        if not os.path.isfile(file):
            continue
        new_filename = os.path.dirname(file) + "/finished/" + os.path.basename(file)
        logger.info(f"Moving {file} to {new_filename}")
        os.renames(file, new_filename)
    return True
