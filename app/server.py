from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from app.config import get_config
from rag.chain_rag import ChainBuilder
from rag.chains import Chains

from logger import logger


app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


chain_builder = ChainBuilder()
chains = Chains({"rag": chain_builder.build(get_config())})


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.get("/config")
async def show_config():
    config = get_config()
    return config


# @app.get("/config/reload")
# async def reload_config():
#     logger.info("Reloading configuration")
#     config = get_config()
#     chains["rag"] = chain_builder.build(config)
#     return config


# Добавляем путь к API
add_routes(app, chains["rag"], path="/chat")


if __name__ == "__main__":
    import uvicorn

    server_config = get_config().get("server", {"host": "0.0.0.0", "port": "8010"})

    uvicorn.run(app, host=server_config["host"], port=server_config["port"])
