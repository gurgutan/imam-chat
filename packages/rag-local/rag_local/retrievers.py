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
from typing import Any, List, Optional
from chromadb import Documents
from langchain_core.documents import Document
from langchain_community.vectorstores import VectorStore, Chroma, FAISS
from chromadb.config import Settings

from logger import logger


class ChromaRetreiverComponent:
    display_name = "ChromaRetreiver"
    description = "Chroma Retreiver embedding models."
    documentation = ""

    def build(
        self,
        embedder: Any,
        documents: Optional[List[Document]] = None,
        search_kwargs: Optional[dict] = None,
        path: str = "./db",
        collection_name: str = "rag-local",
        **kwargs,
    ) -> VectorStore:
        client_settings = Settings(
            is_persistent=True, anonymized_telemetry=False  # , persist_directory=path
        )

        if not os.path.isdir(path):
            store = self.create_store(embedder, documents, path, collection_name)
        else:
            store = Chroma(
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
        path: str = "./db",
        collection_name: str = "rag-local",
        **kwargs,
    ):
        assert documents, "Error: no documents"
        client_settings = Settings(
            is_persistent=True, anonymized_telemetry=False, persist_directory=path
        )
        logger.info("Vector db not found. Building indexes...")
        store = Chroma.from_documents(
            persist_directory=path,
            documents=documents,
            collection_name=collection_name,
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
        path: str = "./db",
        **kwargs,
    ) -> VectorStore:

        if not os.path.isfile(path + "//" + self.index_name + ".faiss"):
            store = self.create_store(embedder, documents, path)
        else:
            store = FAISS.load_local(
                path, embedder, allow_dangerous_deserialization=True
            )
        return store.as_retriever(search_kwargs=search_kwargs)

    def create_store(
        self,
        embedder,
        documents: Optional[List[Document]] = None,
        path: str = "./db",
        **kwargs,
    ):
        assert documents, "Error: no documents"
        logger.info("Vector db not found. Building indexes...")
        store = FAISS.from_documents(documents=documents, embedding=embedder)
        store.save_local(folder_path=path, index_name=self.index_name)
        return store
