# -*- coding: utf-8 -*-
# loaders.py
"""
Module contains adapters classes for text loaders by different providers.
"""

# pylint: disable=no-name-in-module
from typing import Sequence, Union
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders import WebBaseLoader, TextLoader, JSONLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader


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
        self, uri: Union[str, Sequence[str]] = ".", jq_schema: str = "", **kwargs
    ) -> JSONLoader:
        return JSONLoader(file_path=uri, jq_schema=jq_schema)
