FROM python:3.11-slim

# Install tools for building llama-cpp-python
# Comment next line if using other llm provider
RUN apt-get update && apt-get install build-essential -y
RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./


COPY ./package[s] ./packages

COPY ./app ./app

# Documents data
COPY ./sources ./sources

# To build  llama-cpp-python for cuBLAS uncomment next line
# RUN export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
RUN poetry install --no-interaction --no-ansi

# For local llm use in llamacpp, llm need to be loaded to local folder:
# /Path/To/Model/ModelFile.gguf
# Example:
# /imam-chat/models/saiga-mistral-7b/saiga-mistral-q4_K.gguf
# Copy models to image folder
# COPY ./models ./models


EXPOSE 8010

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8010

