# pylint: disable=no-name-in-module
# pylint: disable=unused-import
import os.path
import shutil
from typing import Any, List, Optional
from chromadb import Documents
from langchain_core.documents import Document
from langchain_community.vectorstores import VectorStore, Chroma
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
        path: str = "./chroma",
        rebuild: bool = False,
        **kwargs,
    ) -> VectorStore:
        client_settings = Settings(
            is_persistent=True, anonymized_telemetry=False, persist_directory=path
        )
        # TODO: Create Document check befor adding db and remove next lines (del of db)
        collection_name = "rag-local"
        if os.path.isdir(path) and rebuild:
            try:
                shutil.rmtree(path)
            except OSError as e:
                logger.error("Error: %s - %s", e.filename, e.strerror)

        if not os.path.isdir(path):
            assert documents, "Error: no documents"
            logger.info("Vector db not found. Building indexes...")
            store = Chroma.from_documents(
                persist_directory=path,
                documents=documents,
                collection_name=collection_name,
                embedding=embedder,
                client_settings=client_settings,
            )
        else:
            store = Chroma(
                persist_directory=path,
                collection_name=collection_name,
                embedding_function=embedder,
                client_settings=client_settings,
            )

        return store.as_retriever(search_kwargs=search_kwargs)
