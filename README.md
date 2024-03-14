# imam-chat

API for chatbot using LangChain local or open source components

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

## Launch LangServe

```bash
langchain serve
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
docker run -e -p 8080:8010 imam-chat
```
