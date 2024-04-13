# pylint: disable=no-name-in-module
# pylint: disable=unused-import
import json
import os
from pathlib import Path
from rag.loaders import (
    JSONLoaderComponent,
    QuranJSONLoaderComponent,
    RecursiveUrlLoaderComponent,
    TextLoaderComponent,
    WebBaseLoaderComponent,
)
from langchain_community.document_loaders import WebBaseLoader, TextLoader, JSONLoader
from langchain_core.documents import Document


TEST_JSON_FILENAME = "quran_test.json"
TEST_TEXT_FILENAME = "/quran_test.md"


def test_build_mock_data():
    test_dict = {
        "data": [
            {
                "id": 3,
                "s_n": 1,
                "a_n": 3,
                "text": "Most beneficent, ever-merciful,",
                "tafsir": "The second verse speaks of the Divine quality of mercy, employing two adjectives Rahman and Rahim of which are hyperbolic terms in Arabic, and respectively connote the superabundance and perfection of Divine mercy. The reference to this particular attribute in this situation is perhaps intended to be a reminder of the fact that it is not through any external compulsion or inner need or any kind of necessity whatsoever that Allah has assumed the responsibility of nurturing the whole of His creation, but in response to the demand of His own quality of mercy. If this whole universe did not exist, He would suffer no loss; if it does exist, it is no burden to Him.",
                "source": "Quran.com",
                "name": "Maarif-ul-Quran",
                "language": "English",
                "author": "Mufti Muhammad Shafi",
                "translator": "",
            }
        ]
    }

    if not os.path.isfile(TEST_JSON_FILENAME):
        create_all_folders(TEST_JSON_FILENAME)
        with open(TEST_JSON_FILENAME, mode="w", encoding="utf-8") as f:
            json.dump(test_dict, f, ensure_ascii=False)
    assert True


def test_QuranJSONLoaderComponent():
    # Create test data and save
    test_build_mock_data()
    config = {
        "provider": "quranjsonloader",
        "uri": TEST_JSON_FILENAME,
    }
    loader = QuranJSONLoaderComponent().build(**config)
    assert isinstance(loader, JSONLoader)
    data = loader.load()
    assert isinstance(data[0], Document)
    assert len(data) == 1


def test_JSONLoaderComponent():
    config = {
        "provider": "jsonloader",
        "uri": TEST_JSON_FILENAME,
        "jq_schema": ".data[]",
        "content_key": ".tafsir",
    }
    loader = JSONLoaderComponent().build(**config)
    assert isinstance(loader, JSONLoader)


def test_TextLoaderComponent():
    config = {
        "provider": "textloader",
        "uri": TEST_TEXT_FILENAME,
    }
    loader = TextLoaderComponent().build(**config)
    assert isinstance(loader, TextLoader)


def test_WebBaseLoader():
    """Тест использует ссылки на файлы и на url для загрузки данных"""
    config = {
        "provider": "webbaseloader",
        "uri": "https://ummah.su/info/terminy-islama",
    }
    loader = WebBaseLoaderComponent().build(**config)
    assert isinstance(loader, WebBaseLoader)


def test_RecursiveUrlLoaderComponent():
    pass


def create_all_folders(folder_path):
    """Создает все папки в указанном пути."""
    folder = Path(os.path.dirname(folder_path))
    # Get only folders part of path
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
