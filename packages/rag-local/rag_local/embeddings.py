# Temporary pylint disablings while code under heavy changings
# pylint: disable=no-name-in-module
# pylint: disable=unused-import
from typing import Optional, Dict
from langchain_community.embeddings.huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceBgeEmbeddings,
)
from langchain_community.embeddings import GPT4AllEmbeddings
import torch


class HuggingFaceEmbeddingsComponent:
    display_name = "HuggingFaceEmbeddings"
    description = "HuggingFace sentence_transformers embedding models."
    documentation = "https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html"

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
    display_name = "HuggingFaceBgeEmbeddings"
    description = "HuggingFaceBgeEmbeddings embedding models."
    documentation = "https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceBgeEmbeddings.html"

    def build(
        self,
        cache_folder: Optional[str] = None,
        encode_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
        model_name: str = "BAAI/bge-small-en-v1.5",
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


class GPT4AllEmbeddingsComponent:
    display_name = "GPT4AllEmbeddings"
    description = "GPT4AllEmbeddings embedding models."
    documentation = "https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.gpt4all.GPT4AllEmbeddings.html"

    def build(self, model_name: str = "", **kwargs) -> GPT4AllEmbeddings:
        device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
        return GPT4AllEmbeddings(model_name=model_name, device=device)  # type: ignore
