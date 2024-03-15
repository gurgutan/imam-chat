from typing import Any, Optional
from langchain_community.llms import VLLM, CTransformers, LlamaCpp
from langchain_openai import OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os


DEFAULT_HF_MODEL = "IlyaGusev/saiga_mistral_7b_gguf"
DEFAULT_HF_MODEL_FILE = "saiga_mistral_7b.Q4_0.gguf"
# TheBloke/saiga_mistral_7b-GGUF


class VLLMComponent:
    display_name = "VLLMComponent"
    description = "vLLM model"
    documentation = """To use, you should have the ``vLLM`` python package installed
    https://python.langchain.com/docs/integrations/llms/vllm"""

    def build(
        self,
        model: str,
        max_tokens: int = 256,
        top_k: int = 4,
        top_p: float = 0.95,
        temperature: float = 0.1,
        **kwargs,
    ) -> Any:
        llm = VLLM(
            model=model,
            trust_remote_code=True,  # mandatory for hf models
            max_new_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        return llm


class CTransformersComponent:
    display_name = "CTransformersComponent"
    description = "C Transformers LLM model"
    documentation = """C Transformers LLM models.

    To use, you should have the ``ctransformers`` python package installed.
    See https://python.langchain.com/docs/integrations/llms/ctransformers"""

    def build(
        self,
        model: str,
        model_file: str,
        max_tokens: int = 256,
        top_k: int = 4,
        top_p: float = 0.95,
        temperature: float = 0.1,
        n_ctx: int = 4096,
        **kwargs,
    ) -> Any:
        callbacks = [StreamingStdOutCallbackHandler()]
        full_name = "./models/" + model + "/" + model_file
        model_file = full_name if os.path.isfile(full_name) else None
        context_length = n_ctx
        llm = CTransformers(
            model=model,
            model_file=model_file,
            trust_remote_code=True,  # mandatory for hf models
            max_new_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            context_length=context_length,
            hf=True,
            callbacks=callbacks,
        )
        return llm


class LlamaCppComponent:
    display_name = "LlamaCppComponent"
    description = "LlamaCpp LLM model"
    documentation = """LlamaCpp LLM models.

    To use, you should have the ``llama-cpp-python`` python package installed.
    See https://python.langchain.com/docs/integrations/llms/llamacpp"""

    def build(
        self,
        model: str,
        model_file: str = "",
        max_tokens: int = 256,
        top_k: int = 4,
        top_p: float = 0.95,
        temperature: float = 0.1,
        n_ctx: int = 4096,
        **kwargs,
    ) -> Any:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(
            model_path=model + "/" + model_file,
            callbacks=callback_manager,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            n_ctx=n_ctx,
            verbose=True,  # Verbose is required to pass to the callback manager
        )
        return llm


class OpenAIComponent:
    display_name = "OpenAIChatComponent"
    description = "OpenAI Chat model"
    documentation = """OpenAI Chat models.

    To use, you should have the ``openai`` python package installed.
    See https://python.langchain.com/docs/integrations/llms/openai"""

    def build(
        self,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 256,
        top_k: int = 4,
        top_p: float = 0.95,
        temperature: float = 0.1,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> Any:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = OpenAI(
            model=model,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            verbose=True,
            callbacks=callback_manager,
        )
        return llm
