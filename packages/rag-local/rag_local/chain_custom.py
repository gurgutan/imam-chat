# import os
# from pprint import pprint

from datetime import datetime
import json
from typing import Dict
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
from langchain_core.vectorstores import VectorStoreRetriever
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


class Question(BaseModel):
    __root__: str


class ChainBuilder:
    config: Dict
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
        self.retriever = build_retriever(chunks, embedder, config["vectordb"])
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
            RunnableLambda(self._init_record)
            | RunnableParallel(
                {"context": self.retriever, "question": RunnablePassthrough()}
            )
            | RunnableLambda(self._save_context)
            | self.prompt
            | self.model
            | StrOutputParser()
            | RunnableLambda(self._save_answer)
            | RunnableLambda(self._save_record)
        )

        return rag_chain.with_types(input_type=Question)

    def _init_record(self, question: str):
        self.record = {
            "date": datetime.now(),
            "question": question,
            "prompt_scheme": self.prompt.pretty_repr(),
            "llm_type_emb": self.config["embedder"]["model_name"],
            "llm_type_ans": self.config["llm"]["model_file"],
            "metric_type": self.config["vectordb"]["metric_type"],
            "score": 0.0,
            # TODO: scores
        }
        return question

    def _save_question(self, x):
        self.record["question"] = x
        return x

    def _save_answer(self, x):
        self.record["answer"] = x
        return x

    def _save_context(self, x):
        fragments = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in x["context"]
        ]
        self.record["documents"] = json.dumps(fragments)
        return x

    def _save_prompt(self, x):
        self.record["prompt_scheme"] = self.prompt.pretty_print()
        return x

    def _save_record(self, x):
        logger.info(self.record)
        if not insert(self.connection, self.record):
            logger.error("Failed to insert record to db")
        count_all_raws = count(self.connection)
        logger.info("Records count in db: %i", count_all_raws)
        return x


# TODO: Добавить логирование диалогов
# TODO: Добавить контроль длины промпта
# TODO: Добавить в цепь ConversationBufferMemory
# TODO: Добавить в ответ данные по источникам (метаданные в коллекцию документов и их извелечение)
# TODO: Поискать модель для арабского языка либо зафайнюнить
