from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.prompt_values import (
    PromptValue,
    StringPromptValue,
)

from rag_local.prompts import (
    MuslimImamPrompt,
    QuestionAnswerPrompt,
    QuestionAnswerCoTPrompt,
)


def mock_retreiver(k: int = 1):
    """
    Returns a list of `Document` objects containing the text "Смысл жизни в том чтобы кодить на Python" repeated `k` times.

    Args:
        k (int, optional): The number of `Document` objects to return. Defaults to 1.

    Returns:
        list[Document]: A list of `Document` objects.
    """
    return [Document("Смысл жизни в том чтобы кодить на Python")] * k


def test_QuestionAnswerPrompt():
    """
    Tests the QuestionAnswerPrompt class by invoking it with a context and question,
    and verifying that the returned PromptValue instance has a non-empty string
    representation.
    """
    prompt = QuestionAnswerPrompt().build()
    context = "Смысл жизни в том чтобы кодить на Python"
    question = "В чем смысл жизни?"
    value = prompt.invoke({"context": context, "question": question})
    assert isinstance(value, PromptValue)
    assert len(value.to_string()) > 0


def test_QuestionAnswerCoTPrompt():
    """
    Tests the QuestionAnswerCoTPrompt class, which is used to generate a prompt for a question-answering task.
    The test creates an instance of the QuestionAnswerPrompt class, sets up a context and question, and then invokes the prompt to generate a PromptValue object. It asserts that the generated PromptValue is of the expected type and that its string representation has a length greater than 0.
    """
    prompt = QuestionAnswerPrompt().build()
    context = "Смысл жизни в том чтобы кодить на Python"
    question = "В чем смысл жизни?"
    value = prompt.invoke({"context": context, "question": question})
    assert isinstance(value, PromptValue)
    assert len(value.to_string()) > 0


def test_MuslimImamPrompt():
    """
    Tests the `MuslimImamPrompt` class, which is a prompt template for generating responses from an AI assistant that specializes in providing Islamic guidance and advice.

    The test creates an instance of the `MuslimImamPrompt` class, sets up a context and question, and then invokes the prompt to generate a response. The test asserts that the generated response is a `PromptValue` instance and that the response string has a length greater than 0.
    """
    prompt = MuslimImamPrompt().build()
    context = "Смысл жизни в том чтобы кодить на Python"
    question = "В чем смысл жизни?"
    value = prompt.invoke({"context": context, "question": question})
    assert isinstance(value, PromptValue)
    assert len(value.to_string()) > 0
