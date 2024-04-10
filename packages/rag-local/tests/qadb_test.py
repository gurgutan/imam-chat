import psycopg2
import pytest
from rag_local.qadb import connect_qa_db


@pytest.fixture
def connection():
    connection = psycopg2.connect(
        user="test_user",
        password="test_password",
        host="localhost",
        port=5432,
        database="test_db",
    )
    return connection


def test_connect_qa_db():
    connection = connect_qa_db()
    assert connection is not None, "Connection to qadb failed"
    assert isinstance(connection, psycopg2.extensions.connection)
    connection.close()
