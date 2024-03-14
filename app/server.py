from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_local import build_chain as build_rag_chain
from app.config import get_config

app = FastAPI()

config = get_config()
rag_chain = build_rag_chain(config)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Добавляем путь к API
add_routes(app, rag_chain, path="/chat")

if __name__ == "__main__":
    import uvicorn
    server_config = config.get(
        "server",
        {"host": "0.0.0.0", "port": "8000"}
    )
    uvicorn.run(app, host=server_config["host"], port=server_config["port"])

# TODO: Добавить параметры сервера из config.yml
# TODO: добавить дефолтный конфиг, который мержиться с текущим
