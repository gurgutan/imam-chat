# imam-chat

API for chatbot using LangChain local or open source components. 
Based on LangServe, see https://python.langchain.com/docs/langserve.

API endpoints:
  - **http://host:8010/docs**  - generated OpenAPI docs
  - **http://host:8010/chat**  - main dialog endpoint
  - **http://host:8010/chat/playground** - playground for chat-bot

## Installation
Install poetry if you haven't yet (see https://python-poetry.org/docs/#installing-with-the-official-installer)

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
Supported models in format of choosen provider (see config.yml comments in 'llm' property).
For local LLM it need to be loaded to folder '/models/'.
For example, https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF.
Example of path to file: /imam-chat/models/saiga-mistral-7b/saiga-mistral-q4_K.gguf
You can choose model that will be use in config.yml/llm section

## Settings
All editable settings in app/config.yml. See comments in config.yml.

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

When building the image, llms files from folder /models/ copied. You can comment this lines if provider is ctransformers to load models from hugging face in app first start.
See dockerfile comments.

### Running the Image Locally

Run the image with folder ./db binded to folder /code/db/ and ./models binded to /code/models/. This folder contains vector stores of documents with indexes.
```shell
docker run --rm -it -p 8010:8010/tcp -v ./db:/code/db -v ./models:/code/models imam-chat:latest 
```

### Build & Run Image
```shell
docker build . -t imam-chat && \
docker run --rm -it -p 8010:8010/tcp -v ./db:/code/db -v ./models:/code/models imam-chat:latest 
```


## Examples or questions
```
What is Five Pillars of Islam?
```
```
How many surahs are there in the Holy Quran?
```
```
Can i have 6 wifes?
```

# Issues

## 1. Poetry requires python 3.11, but there is only python3.10 on your machine
Install python 3.11
#### Step 1. Update & upgrade
```bash
sudo apt update
sudo apt upgrade
```

#### Step 2: Import Python PPA on Ubuntu 22.04 or 20.04
```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
```

#### Step 3: Install Python 3.11 on Ubuntu 22.04 or 20.04
```bash
sudo apt install python3.11
python3.11 --version
```
Debug module:
```bash
sudo apt install python3.11-dbg
```
Developer module:
```bash
sudo apt install python3.11-dev
```

#### Step 4: Install PIP with Python 3.11 on Ubuntu 22.04 or 20.04
```bash
sudo apt install python3-pip
```
Check for updates:
```bash
python3.11 -m pip install --upgrade pip
pip --version
```

#### Step 5. Switch Default Python Versions on Ubuntu
```bash
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 9
```
Last number represent priority of the version. A higher number means hihger priority. So 9 - means maximum priority if installed less then 9 python versions.

Switch to python 3.11
```bash
sudo update-alternatives --config python
```
## Issue 2. langserv on start raise ERROR: '[Errno 98] Address already in use'
Uncorrect stop of server process keep listening on the port.
To find process use 
```bash
lsof -i :8010
```

To kill use
```bash
sudo lsof -t -i tcp:8010 | xargs kill -9
```


## TODO
1. Add providers for auto download models (HF, for example)/
2. Add context window managment (length, queue).
3. Create Quran DB scheme.
4. 
