import psycopg2
import pytest
from rag.qadb import connect_qa_db


@pytest.fixture
def connection():
    conn = psycopg2.connect(
        user="test_user",
        password="test_password",
        host="localhost",
        port=5432,
        database="test_db",
    )
    return conn


def test_connect_qa_db():
    conn = connect_qa_db()
    assert conn is not None, "Connection to qadb failed"
    assert isinstance(conn, psycopg2.extensions.connection)
    conn.close()
