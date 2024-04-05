# pylint: disable=no-name-in-module
# pylint: disable=unused-import
from typing import Any, List, Optional
from chromadb import Documents
from langchain_core.documents import Document
from langchain_community.vectorstores import VectorStore, Chroma
from chromadb.config import Settings


class ChromaRetreiverComponent:
    display_name = "ChromaRetreiver"
    description = "Chroma Retreiver embedding models."
    documentation = ""

    def build(
        self,
        embedder: Any,
        documents: Optional[List[Document]] = None,
        search_kwargs: Optional[dict] = None,
        **kwargs
    ) -> VectorStore:
        client_settings = Settings(anonymized_telemetry=False)
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
