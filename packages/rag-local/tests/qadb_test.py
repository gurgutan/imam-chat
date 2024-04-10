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
    assert connect_qa_db() is not None

    assert connection is not None


def test_connect_qa_db_error(monkeypatch):
    def raise_error(*args, **kwargs):
        raise psycopg2.OperationalError

    monkeypatch.setattr(psycopg2, "connect", raise_error)
    with pytest.raises(psycopg2.OperationalError):
        connect_qa_db()
