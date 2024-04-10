from typing import List
from langchain_text_splitters import CharacterTextSplitter


class QuranTextSplitter(CharacterTextSplitter):

    def split_text(self, text: str) -> List[str]:
        chunks = super().split_text(text)
        return chunks
