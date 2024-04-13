# Temporary pylint disablings while code under heavy changings
# pylint: disable=no-name-in-module
# pylint: disable=unused-import
from typing import Optional, Dict
from langchain_community.embeddings.huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceBgeEmbeddings,
)
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from rag.component import raise_not_implemented
import torch


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
        "huggingfacebgeembedding": HuggingFaceBgeEmbeddingsComponent().build,
    }
    embedder = providers.get(config["provider"].lower(), raise_not_implemented)(
        **config
    )
    return embedder


class HuggingFaceEmbeddingsComponent:
    description = "HuggingFace sentence_transformers embedding models."
    documentation = (
        "https://python.langchain.com/docs/integrations/platforms/huggingface/"
    )

    def build(
        self,
        cache_folder: Optional[str] = None,
        encode_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        **kwargs
    ) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            cache_folder=cache_folder,
            encode_kwargs=encode_kwargs,
            model_kwargs=model_kwargs,
            model_name=model_name,
            multi_process=False,
            show_progress=True,
        )


class HuggingFaceBgeEmbeddingsComponent:
    description = "HuggingFaceBgeEmbeddings embedding models."
    documentation = "https://python.langchain.com/docs/integrations/platforms/huggingface/#huggingfacebgeembeddings"

    def build(
        self,
        cache_folder: Optional[str] = None,
        encode_kwargs: Optional[Dict] = {"normalize_embeddings": True},
        model_kwargs: Optional[Dict] = {"device": "cpu", "trust_remote_code": True},
        model_name: str = "BAAI/bge-small-en-v1.5",
        **kwargs
    ) -> HuggingFaceBgeEmbeddings:
        return HuggingFaceBgeEmbeddings(
            cache_folder=cache_folder,
            encode_kwargs=encode_kwargs,
            model_kwargs=model_kwargs,
            model_name=model_name,
        )


class GPT4AllEmbeddingsComponent:
    description = "GPT4AllEmbeddings embedding models."
    documentation = (
        "https://python.langchain.com/docs/integrations/text_embedding/gpt4all/"
    )

    def build(self, model_name: str = "", **kwargs) -> GPT4AllEmbeddings:
        device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
        return GPT4AllEmbeddings(model_name=model_name, device=device)  # type: ignore


class OpenAIEmbeddingsComponent:
    description = "OpenAIEmbeddings embedding models."
    documentation = (
        "https://python.langchain.com/docs/integrations/text_embedding/openai/"
    )

    def build(
        self,
        model_name: str = "text-embedding-3-large",
        base_url: str | None = "https://api.openai.com/v1/",
        api_key: SecretStr | None = None,
        **kwargs
    ) -> OpenAIEmbeddings:
        if model_name == "":
            model_name = "text-embedding-ada-002"
        return OpenAIEmbeddings(model=model_name, base_url=base_url, api_key=api_key)
