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

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_TABLE_NAME = os.getenv("DB_TABLE_NAME")

# CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# Scheme
# CREATE TABLE public.query_response (
#     id bigint NOT NULL,
#     date timestamp without time zone,
#     question text,
#     answer text,
#     documents json,
#     scores real,
#     metric_type text,
#     llm_type_emb text,
#     prompt_scheme text,
#     llm_type_ans text
# );


def connect_qa_db() -> PSConnection:
    """Connect to the PostgreSQL database server."""
    try:
        connection = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
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
        data["metric_type"],
        data["llm_type_emb"],
        data["llm_type_ans"],
        data["prompt_scheme"],
    )

    connection.autocommit = True

    # Выполнение SQL-запроса для вставки данных в таблицу
    insert_query = f"""
        INSERT INTO {DB_TABLE_NAME} (date, question, answer, documents, metric_type, llm_type_emb, llm_type_ans, prompt_scheme) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
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
            select_query = f"""SELECT * FROM {DB_TABLE_NAME}"""
            cursor.execute(select_query)
        except (PostgresError, psycopg2.DatabaseError):
            logger.error("Error while selecting data from the table")
            return []

        return cursor.fetchall()


def count(connection: PSConnection) -> int:
    """Count all rows from the table."""
    with connection.cursor() as cursor:
        try:
            select_query = f"""SELECT COUNT(*) FROM {DB_TABLE_NAME}"""
            cursor.execute(select_query)
        except (PostgresError, psycopg2.DatabaseError):
            logger.error("Error while selecting data from the table")
            return 0
        return cursor.fetchone()[0]  # type: ignore
