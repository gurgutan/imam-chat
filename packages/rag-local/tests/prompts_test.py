import pytest

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.prompt_values import (
    PromptValue,
    StringPromptValue,
)

from rag_local.prompts import QuestionAnswerPrompt, QuestionAnswerCoTPrompt


def mock_retreiver(k: int = 1):
    return [Document("Смысл жизни в том чтобы кодить на Python")] * k


def test_QuestionAnswerPrompt():
    prompt = QuestionAnswerPrompt().build()
    context = "Смысл жизни в том чтобы кодить на Python"
    question = "В чем смысл жизни?"
    value = prompt.invoke({"context": context, "question": question})
    assert isinstance(value, PromptValue)
    assert len(value.to_string()) > 0


def test_QuestionAnswerCoTPrompt():
    prompt = QuestionAnswerPrompt().build()
    context = "Смысл жизни в том чтобы кодить на Python"
    question = "В чем смысл жизни?"
    value = prompt.invoke({"context": context, "question": question})
    assert isinstance(value, PromptValue)
    assert len(value.to_string()) > 0
