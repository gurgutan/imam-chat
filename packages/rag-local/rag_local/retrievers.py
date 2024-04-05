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
        # TODO: Create Document check befor adding db and remove next lines (del of db)
        if os.path.isdir(path) and rebuild:
            try:
                shutil.rmtree(path)
            except OSError as e:
                logger.error("Error: %s - %s", e.filename, e.strerror)
        client_settings = Settings(
            is_persistent=True, anonymized_telemetry=False, persist_directory=path
        )
        if documents:
            return Chroma.from_documents(
                documents=documents,
                collection_name="rag-local",
                embedding=embedder,
                client_settings=client_settings,
            ).as_retriever(search_kwargs=search_kwargs)
        # TODO: from_db
        else:
            raise NotImplementedError("Not implemented persistent vectorstore")
