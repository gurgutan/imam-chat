
# rag-local

!!!Not ready yet.

This performs RAG with no reliance on externallocal APIs. 

Settings and API provider can be set in app/config.yml

It utilizes LlamaCpp | CTransformers the LLM, GPT4All | HuggingFaceEmbeddings for embeddings, and Chroma for the vectorstore.

The vectorstore is created in `chain.py` and by default indexes a [popular islam termis site](https://ummah.su/info/terminy-islama/) for question-answering.
Other type of data sources can be set in app/config.yml

## Environment Setup

To set up the environment, you need to install LlamaCpp | CTransformers. 

To use CTransformers, you should have the ``ctransformers`` python package installed.
See https://python.langchain.com/docs/integrations/llms/ctransformers

To use LlamaCPP, you should have the ``llama-cpp-python`` python package installed.
See https://python.langchain.com/docs/integrations/llms/llamacpp"

This package also uses [GPT4All](https://python.langchain.com/docs/integrations/text_embedding/gpt4all) embeddings. 

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-local
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-local
```

And add the following code to your `server.py` file:
```python
from rag_local import build_chain as build_rag_chain

add_routes(app, build_rag_chain, path="/chat")
```

(Optional) Let's now configure LangSmith. LangSmith will help us trace, monitor and debug LangChain applications. LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/). If you don't have access, you can skip this section

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve --port 8010
```

This will start the FastAPI app with a server is running locally at 
[http://localhost:8010](http://localhost:8010)

We can see all templates at [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs)
We can access the playground at [http://127.0.0.1:8010/chat/playground](http://127.0.0.1:8010/chat/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8010/chat")
```


