# -*- coding: utf-8 -*-
# llms.py
"""
Module contains adapter classes for instantiating LLM by different providers.
"""

# Temporary pylint disablings while code under heavy changings
# pylint: disable=no-name-in-module
# pylint: disable=unused-import
import os
import os.path

from typing import Any, Dict, Optional
from langchain_community.llms import VLLM, VLLMOpenAI, CTransformers, LlamaCpp, Ollama
from langchain_community.chat_models import ChatOpenAI

# from langchain_openai import OpenAI

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import SecretStr
from rag.component import raise_not_implemented


DEFAULT_BASE_URL = "http://ollama:11434"  # ollama local host


def build_model(config: Dict):
    """Build the model based on the config dict

    Args:
        config (Dict): dictionary with the following keys:
            provider (str): LlamaCpp | CTransformers | VLLM
            model_name (str) : path to the model or name of the model in huggingface hub

    Returns:
        model

    Raises:
        Exception: NotImplementedError if unknown provider
    """
    providers = {
        "ollama": OllamaComponent().build,
        "llamacpp": LlamaCppComponent().build,
        "ctransformers": CTransformersComponent().build,
        "vllm": VLLMComponent().build,
        "vllmopenai": VLLMOpenAIComponent().build,
        "openai": OpenAIComponent().build,
    }
    model = providers.get(config["provider"].lower(), raise_not_implemented)(**config)
    return model


class OllamaComponent:
    description = "Ollama API service"
    documentation = """You should have the ``Ollama`` service running
    https://python.langchain.com/docs/integrations/llms/ollama/
    """

    def build(
        self,
        model: str = "llama2",
        base_url: str = DEFAULT_BASE_URL,
        n_ctx: Optional[int] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        threads: Optional[int] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> Any:
        llm = Ollama(
            model=model,
            base_url=base_url,
            num_ctx=n_ctx,
            num_predict=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_thread=threads,
            system=system,
            keep_alive="12h",
        )
        return llm


class VLLMComponent:
    description = "vLLM model"
    documentation = """You should have the ``vLLM`` python package installed
    https://python.langchain.com/docs/integrations/llms/vllm"""

    def build(
        self,
        model: str,
        max_tokens: int = 2048,
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
    description = "C Transformers LLM model"
    documentation = """C Transformers LLM models.

    To use, you should have the ``ctransformers`` python package installed.
    See https://python.langchain.com/docs/integrations/llms/ctransformers"""

    def build(
        self,
        model: str,
        model_file: str,
        max_tokens: int = 2048,
        top_k: int = 4,
        top_p: float = 0.95,
        temperature: float = 0.1,
        n_ctx: int = 4096,
        threads: int = 4,
        **kwargs,
    ) -> Any:
        callbacks = [StreamingStdOutCallbackHandler()]
        # full_name = os.path.join(model, model_file)
        # model_file = full_name if os.path.isfile(full_name) else model_file
        config = {
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "context_length": n_ctx,
            "threads": threads,
            "max_new_tokens": max_tokens,
        }
        llm = CTransformers(
            model=model,
            model_file=model_file,
            trust_remote_code=True,  # mandatory for hf models
            config=config,
            hf=True,
            callbacks=callbacks,
        )
        return llm


class LlamaCppComponent:
    description = "LlamaCpp LLM model"
    documentation = """LlamaCpp LLM models.

    To use, you should have the ``llama-cpp-python`` python package installed.
    See https://python.langchain.com/docs/integrations/llms/llamacpp"""

    def build(
        self,
        model: str,
        model_file: str = "",
        max_tokens: int = 2048,
        top_k: int = 4,
        top_p: float = 0.95,
        temperature: float = 0.1,
        n_ctx: int = 4096,
        threads: int = 4,
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
            n_threads=threads,
            verbose=True,  # required to pass to the callback manager
        )
        return llm


class OpenAIComponent:
    description = "OpenAI Chat model"
    documentation = """OpenAI Chat models.

    To use, you should have the ``openai`` python package installed.
    See https://python.langchain.com/docs/integrations/llms/openai"""

    def build(
        self,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 2048,
        top_p: float = 0.95,
        temperature: float = 0.1,
        base_url: Optional[str] = None,
        api_key: Optional[SecretStr | None] = None,
        **kwargs,
    ) -> Any:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            verbose=True,
            callbacks=callback_manager,
        )
        return llm


class VLLMOpenAIComponent:
    description = "OpenAI mimic chat model"
    documentation = """OpenAI Server protocol chat models."""

    def build(
        self,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 2048,
        top_p: float = 0.95,
        temperature: float = 0.1,
        base_url: Optional[str] = None,
        api_key: Optional[SecretStr | None] = None,
        **kwargs,
    ) -> Any:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = VLLMOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            verbose=True,
            callbacks=callback_manager,
        )
        return llm
