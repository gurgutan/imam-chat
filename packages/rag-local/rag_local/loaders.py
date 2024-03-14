from typing import Sequence, Union
from langchain_community.document_loaders import WebBaseLoader, TextLoader, JSONLoader


# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()


class WebBaseLoaderComponent:
    display_name = "WebBaseLoaderComponent"
    description = "Web Loader Component"
    documentation = ""

    def build(self, uri: Union[str, Sequence[str]] = "", **kwargs) -> WebBaseLoader:
        return WebBaseLoader(uri)


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
