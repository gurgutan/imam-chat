from typing import Dict, Iterable
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)


class Chains:
    def __init__(self, chains: Dict[str, RunnableLambda]):
        self.__chains = chains.copy()

    def __getitem__(self, chain_name: str):
        return self.__chains.get(chain_name, None)

    def __setitem__(self, chain_name: str, runnable: RunnableLambda):
        self.__chains[chain_name] = runnable

    def get_chains(self):
        return self.__chains
