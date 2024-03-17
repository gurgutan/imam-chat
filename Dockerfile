FROM python:3.11-slim

# Установка средств для компиляции (нужно для llama-cpp-python)
RUN apt-get update && apt-get install build-essential -y
RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./package[s] ./packages

RUN poetry install --no-interaction --no-ansi --no-root

COPY ./app ./app

# Данные документов
COPY ./data ./data

# Для использования локальной модели через llamacpp, модель нужно загрузить в образ
# Структура папок должна быть следующей:
# /Path/To/Model/ModelFile.gguf
# Пример:
# /imam-chat/models/saiga-mistral-7b/saiga-mistral-q4_K.gguf
# Копирование модели (нужно отключить на проде)
COPY ./models ./models


# Для сборки llama-cpp-python под cuBLAS раскомментировать следующую строку
# RUN export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
RUN poetry install --no-interaction --no-ansi

EXPOSE 8010

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8010

# TODO: подключать том с моделями?
