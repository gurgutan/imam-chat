# -*- coding: utf-8 -*-
# llms.py
"""
Module contains adapter classes for vectore store with retriever from different providers.
"""

# Temporary pylint disablings while code under heavy changings
# pylint: disable=no-name-in-module
# pylint: disable=unused-import
import os.path
import shutil
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from langchain_community.vectorstores import VectorStore
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.chroma import Chroma


from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)

from chromadb.config import Settings

from logger import logger
from rag_local.component import raise_not_implemented

# Путь по умолчанию к векторной БД
DEFAULT_STORE_PATH = "./store"


def build_retriever(documents, embedder, config: Dict) -> VectorStoreRetriever:
    """Build the retriever based on the config dict"""
    providers = {
        "chroma": ChromaRetreiverComponent().build,
        "faiss": FAISSRetreiverComponent().build,
    }
    retriever = providers.get(config["provider"].lower(), raise_not_implemented)(
        documents=documents, embedder=embedder, **config
    )
    return retriever


class ChromaRetreiverComponent:
    display_name = "ChromaRetreiver"
    description = "Chroma Retreiver embedding models."
    documentation = ""
    distance_strategy = DistanceStrategy.COSINE

    def build(
        self,
        embedder: Any,
        documents: Optional[List[Document]] = None,
        search_kwargs: Optional[dict] = None,
        path: str = DEFAULT_STORE_PATH,
        collection_name: str = "rag-local",
        metric_type: str = "cosine",
        **kwargs,
    ) -> VectorStoreRetriever:
        client_settings = Settings(
            is_persistent=True, anonymized_telemetry=False  # , persist_directory=path
        )
        if metric_type not in ["cosine", "l2", "ip"]:
            raise ValueError(
                f"{metric_type} is not supported. Please use one of ['cosine', 'l2', 'ip']"
            )

        metadata = {"hnsw:space": metric_type}

        if not os.path.isdir(path):
            logger.info(f"Vector db not found in {path}. Building indexes...")
            store: Chroma = self.create_store(
                embedder, documents, path, collection_name, metadata
            )
        else:
            store: Chroma = Chroma(
                persist_directory=path,
                collection_name=collection_name,
                embedding_function=embedder,
                client_settings=client_settings,
            )

        return store.as_retriever(search_kwargs=search_kwargs)

    def create_store(
        self,
        embedder,
        documents: Optional[List[Document]] = None,
        path: str = DEFAULT_STORE_PATH,
        collection_name: str = "rag-local",
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> Chroma:
        if not documents:
            raise ValueError("Error: no documents for create_store")
        client_settings = Settings(
            is_persistent=True, anonymized_telemetry=False, persist_directory=path
        )

        store: Chroma = Chroma.from_documents(
            persist_directory=path,
            documents=documents,
            collection_name=collection_name,
            collection_metadata=metadata,
            embedding=embedder,
            client_settings=client_settings,
        )
        return store


class FAISSRetreiverComponent:
    display_name = "FAISSRetreiver"
    description = "FAISS Retreiver embedding models."
    documentation = ""
    index_name = "index"

    def build(
        self,
        embedder: Any,
        documents: Optional[List[Document]] = None,
        search_kwargs: Optional[dict] = None,
        path: str = DEFAULT_STORE_PATH,
        metric_type: str = "cosine",
        **kwargs,
    ) -> VectorStoreRetriever:

        if not os.path.isfile(path + "//" + self.index_name + ".faiss"):
            logger.info(f"Vector db not found in {path}. Building indexes...")
            store: FAISS = self.create_store(embedder, documents, path)
        else:
            store = FAISS.load_local(
                path, embedder, allow_dangerous_deserialization=True
            )
        metrics_map = {
            "cosine": DistanceStrategy.COSINE,
            "l2": DistanceStrategy.EUCLIDEAN_DISTANCE,
            "ip": DistanceStrategy.MAX_INNER_PRODUCT,
            "jaccard": DistanceStrategy.JACCARD,
        }
        store.distance_strategy = metrics_map.get(
            metric_type.lower(), DistanceStrategy.COSINE
        )
        # Also, we can specify distance strategy
        # DistanceStrategy.MAX_INNER_PRODUCT:
        # DistanceStrategy.EUCLIDEAN_DISTANCE:
        return store.as_retriever(search_kwargs=search_kwargs)

    def create_store(
        self,
        embedder,
        documents: Optional[List[Document]] = None,
        path: str = DEFAULT_STORE_PATH,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        **kwargs,
    ) -> FAISS:
        if not documents:
            raise ValueError("Error: no documents for create_store")

        store = FAISS.from_documents(documents=documents, embedding=embedder)
        store.distance_strategy = distance_strategy
        store.save_local(folder_path=path, index_name=self.index_name)
        return store
