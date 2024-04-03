# imam-chat

API for chatbot using LangChain local or open source components. 


## Installation
Install poetry if you haven't yet

```bash
pip install poetry
```

Install the project dependencies

```bash
poetry install
```

Install the LangChain CLI if you haven't yet

```bash
pip install -U langchain-cli
```

## Load models
Load LLM to folder '/models/'. Supported models in gguf format.
For example, https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF.
Example of path to file: /imam-chat/models/saiga-mistral-7b/saiga-mistral-q4_K.gguf
You can choose model that will be use in config.yml/llm section

## Settings
All editable settings in app/config.yml. See comments in file.

## Launch LangServe

```bash
langchain serve --port 8010
```

## Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

### Building the Image

To build the image, you simply:

```shell
docker build . -t imam-chat
```


### Running the Image Locally

```shell
docker run -e -p 8010:8010 imam-chat
```
