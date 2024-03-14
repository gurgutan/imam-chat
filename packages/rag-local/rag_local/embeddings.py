
from typing import Optional, Dict
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings


class HuggingFaceEmbeddingsComponent():
    display_name = "HuggingFaceEmbeddings"
    description = "HuggingFace sentence_transformers embedding models."
    documentation = (
        "https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html"
    )

    def build(
        self,
        cache_folder: Optional[str] = None,
        encode_kwargs: Optional[Dict] = {},
        model_kwargs: Optional[Dict] = {},
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        multi_process: bool = False,
    ) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            cache_folder=cache_folder,
            encode_kwargs=encode_kwargs,
            model_kwargs=model_kwargs,
            model_name=model_name,
            multi_process=multi_process,
        )


class GPT4AllEmbeddingsComponent():
    display_name = "GPT4AllEmbeddings"
    description = "GPT4AllEmbeddings embedding models."
    documentation = (
        "https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.gpt4all.GPT4AllEmbeddings.html"
    )

    def build(
        self,
        model_name: Optional[str] = None,
        n_threads: Optional[int] = None
    ) -> GPT4AllEmbeddings:
        return GPT4AllEmbeddings(model_name=model_name, n_threads=n_threads)
