"""
Создание базы данных для работы с Q&A
"""

import os
from datetime import datetime
from logger import logger
from dotenv import load_dotenv, find_dotenv
from typing import Dict
import psycopg2
from psycopg2 import Error as PostgresError
from psycopg2.extensions import connection as PSConnection

load_dotenv(find_dotenv())

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_PORT = "5432"
POSTGRES_HOST = "localhost"
DB_NAME = "imam"


# CONNECTION_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{DB_NAME}"
TABLE_NAME = "query_response"

# Scheme
# CREATE TABLE public.query_response (
#     id bigint NOT NULL,
#     date timestamp without time zone,
#     question text,
#     answer text,
#     documents json,
#     score real,
#     metric_type text,
#     llm_type_emb text,
#     prompt_scheme text,
#     llm_type_ans text
# );


def connect_qa_db() -> PSConnection:
    """Connect to the PostgreSQL database server."""
    try:
        connection = psycopg2.connect(
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=DB_NAME,
        )
    except PostgresError as error:
        logger.error("Unable to connect database: %s", error)
        raise PostgresError(error)

    return connection


def insert(connection: PSConnection, data: Dict) -> bool:
    """
    Insert data into the table
    """
    record = (
        data["date"],
        data["question"],
        data["answer"],
        data["documents"],
        data["score"],
        data["metric_type"],
        data["llm_type_emb"],
        data["llm_type_ans"],
        data["prompt_scheme"],
    )

    connection.autocommit = True

    # Выполнение SQL-запроса для вставки данных в таблицу
    insert_query = f"""
        INSERT INTO {TABLE_NAME} (date, question, answer, documents, score, metric_type, llm_type_emb, llm_type_ans, prompt_scheme) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
    with connection.cursor() as cursor:
        try:
            cursor.execute(insert_query, record)
        except (TypeError, psycopg2.DatabaseError) as error:
            logger.error(error)
            return False
    # connection.commit()
    return True


def select(connection: PSConnection) -> list:
    """Select all rows from the table."""

    with connection.cursor() as cursor:
        try:
            select_query = f"""SELECT * FROM {TABLE_NAME}"""
            cursor.execute(select_query)
        except (PostgresError, psycopg2.DatabaseError):
            logger.error("Error while selecting data from the table")
            return []

        return cursor.fetchall()


def count(connection: PSConnection) -> int:
    """Count all rows from the table."""
    with connection.cursor() as cursor:
        try:
            select_query = f"""SELECT COUNT(*) FROM {TABLE_NAME}"""
            cursor.execute(select_query)
        except (PostgresError, psycopg2.DatabaseError):
            logger.error("Error while selecting data from the table")
            return 0
        return cursor.fetchone()[0]  # type: ignore
