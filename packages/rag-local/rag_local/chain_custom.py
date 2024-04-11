# import os
# from pprint import pprint

from datetime import datetime
import json
from json import encoder
from typing import Any, Dict
from logger import logger

# from langchain_core.prompts import ChatPromptTemplate
# pylint: disable=no-name-in-module
# pylint: disable=unused-import
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import BaseLLM

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_local.llms import build_model
from rag_local.embeddings import build_embedder
from rag_local.loaders import build_loader
from rag_local.retrievers import build_retriever
from rag_local.prompts import MuslimImamPrompt
from rag_local.qadb import connect_qa_db, count, insert, select

from psycopg2.extensions import connection as PSConnection

from rag_local.utils import first, second


class Question(BaseModel):
    __root__: str


class ChainBuilder:
    config: Dict
    store: VectorStore
    retriever: VectorStoreRetriever
    prompt: ChatPromptTemplate
    model: BaseLLM
    embedder: BaseLLM
    connection: PSConnection
    record: Dict = {}

    def build(self, config: Dict):
        """
        Build rag chain from config parameters
        """
        self.config = config
        self.connection = connect_qa_db()
        if not self.connection:
            logger.error("Could not connect to QA DB")
            raise IOError("Could not connect to QA DB")

        # Выбираем загрузчик данных
        loader = build_loader(config["loader"])
        logger.info("Using loader %s", loader.__class__.__name__)

        # Загружаем данные из источника
        logger.info(f"Loading data with settings: {config['loader']} ...")
        data = loader.load()

        # Применяем сплиттер
        logger.info("Documents splitting...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100
        )
        chunks = text_splitter.split_documents(data)
        logger.info("Produced %i chunks", len(chunks))

        # Выбираем эмбеддер
        embedder = build_embedder(config["embedder"])
        logger.info("Using embedder %s", embedder.__class__.__name__)

        # Добавляем vectorstore retriever
        logger.info("Creating data indexes with embedder")
        self.retriever, self.store = build_retriever(
            chunks, embedder, config["vectordb"]
        )
        logger.info("Using retriever %s", self.retriever.__class__.__name__)

        # Prompt
        # prompt = ChatPromptTemplate.from_template(template)
        # prompt = QuestionAnswerCoTPrompt().build()
        self.prompt = MuslimImamPrompt().build()

        # Готовим модель
        logger.info(f"Initiating LLM: {config['llm']}")
        self.model = build_model(config["llm"])

        # Создаем RAG chain
        rag_chain = (
            RunnableLambda(self._init_question)  # -> question
            | RunnableParallel(
                {
                    "docs_with_scores": RunnableLambda(self._get_docs_with_scores),
                    "question": RunnablePassthrough(),
                }
            )
            | RunnableLambda(self._get_context_with_question)
            | self.prompt
            | self.model
            | StrOutputParser()
            | RunnableLambda(self._get_answer)
            | RunnableLambda(self._finilize_chain)
        )

        return rag_chain.with_types(input_type=Question)

    def _init_question(self, question: str):
        # Recording to buffer
        self.record = {
            "date": datetime.now(),
            "question": question,
            "prompt_scheme": self.prompt.pretty_repr(),
            "llm_type_emb": self.config["embedder"]["model_name"],
            "llm_type_ans": self.config["llm"]["model_file"],
            "metric_type": self.config["vectordb"]["metric_type"],
        }
        # Here may be formatting or transformin question
        return question.strip()

    def _get_context_with_question(self, x) -> dict[str, str]:
        """
        Get context string from the
            { "docs_with_scores": docs_with_scores, "question": question }
        to
            {"context": context, "question": question}
        """
        question = x["question"]
        docs_with_scores = x["docs_with_scores"]
        documents = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
            }
            for (doc, score) in zip(
                docs_with_scores["documents"], docs_with_scores["scores"]
            )
        ]
        context = "\nFound quotes:\n"
        context += "\n----\n".join(
            [f"{doc['content']} [{str(i+1)}]" for i, doc in enumerate(documents)]
        )

        quoting = "\n\n"
        quoting += "\n".join(
            [
                f"{i+1}. {self._format_quoting(doc['metadata'])}"
                for i, doc in enumerate(documents)
            ]
        )
        # Save to buffer
        self.record["documents"] = json.dumps(documents, ensure_ascii=False)
        # self.record["scores"] = json.dumps(docs_with_scores["scores"], ensure_ascii=False)
        return {"context": context + quoting, "question": question}

    def _format_quoting(self, metadata: dict) -> str:
        quote_data = [
            metadata.get("author", ""),
            metadata.get("name", ""),
            metadata.get("source", ""),
        ]
        quote = ". ".join([s for s in quote_data if s])
        return quote

    def _get_answer(self, x):
        self.record["answer"] = x
        return x

    def _finilize_chain(self, x):
        logger.info(self.record)
        if not insert(self.connection, self.record):
            logger.error("Failed to insert record to db")
        count_all_raws = count(self.connection)
        logger.info("Records count in db: %i", count_all_raws)
        return x

    def _get_docs_with_scores(self, query: str):
        # retrive collection of type List[Tuple[Document, float]]
        pairs = self.store.similarity_search_with_score(query)
        docs, scores = zip(*pairs)
        return {"documents": docs, "scores": [float(score) for score in scores]}


# TODO: Добавить логирование диалогов
# TODO: Добавить контроль длины промпта
# TODO: Добавить в цепь ConversationBufferMemory
# TODO: Добавить в ответ данные по источникам (метаданные в коллекцию документов и их извелечение)
# TODO: Поискать модель для арабского языка либо зафайнюнить
