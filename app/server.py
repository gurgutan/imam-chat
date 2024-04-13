from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

from app.config import get_config
from rag.chain_custom import ChainBuilder

app = FastAPI()

config = get_config()


# rag_chain = build_chain(config)
chain_builder = ChainBuilder()
rag_chain = chain_builder.build(config)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Добавляем путь к API
add_routes(app, rag_chain, path="/chat")

if __name__ == "__main__":
    import uvicorn

    server_config = config.get("server", {"host": "0.0.0.0", "port": "8010"})

    uvicorn.run(app, host=server_config["host"], port=server_config["port"])
