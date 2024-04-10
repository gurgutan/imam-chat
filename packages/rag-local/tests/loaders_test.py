# pylint: disable=no-name-in-module
# pylint: disable=unused-import
import json
import os
from pathlib import Path
from rag_local.loaders import (
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


def test_build_mock_data(surah: int = 1, ayat: int = 1):
    test_dict = {
        "data": [
            {
                "id": ayat,
                "arabic": "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
                "tafsir_ar": '( بسم الله الرحمن الرحيم ) بسم الله الباء أداة تخفض ما بعدها مثل من وعن والمتعلق به الباء محذوف لدلالة الكلام عليه تقديره أبدأ بسم الله أو قل بسم الله . وأسقطت الألف من الاسم طلبا للخفة وكثرة استعمالها وطولت الباء قال القتيبي ليكون افتتاح كلام كتاب الله بحرف معظم كان عمر بن عبد العزيز رحمه الله يقول لكتابه طولوا الباء وأظهروا السين وفرجوا بينهما ودوروا الميم . تعظيما لكتاب الله تعالى وقيل لما أسقطوا الألف ردوا طول الألف على الباء ليكون دالا على سقوط الألف ألا ترى أنه لما كتبت الألف في " اقرأ باسم ربك " ( 1 - العلق ) ردت الباء إلى صيغتها ولا تحذف الألف إذا أضيف الاسم إلى غير الله ولا مع غير الباء .والاسم هو المسمى وعينه وذاته قال الله تعالى : " إنا نبشرك بغلام اسمه يحيى " ( 7 - مريم ) أخبر أن اسمه يحيى ثم نادى الاسم فقال : يا يحيى " وقال : ما تعبدون من دونه إلا أسماء سميتموها " ( 40 - يوسف ) وأراد الأشخاص المعبودة لأنهم كانوا يعبدون المسميات وقال : سبح اسم ربك " ( 1 - الأعلى ) ، وتبارك اسم ربك " ثم يقال للتسمية أيضا اسم فاستعماله في التسمية أكثر من المسمى فإن قيل ما معنى التسمية من الله لنفسه؟ قيل هو تعليم للعباد كيف يفتتحون القراءة .واختلفوا في اشتقاقه قال المبرد من البصريين هو مشتق من السمو وهو العلو فكأنه علا على معناه وظهر عليه وصار معناه تحته وقال ثعلب من الكوفيين : هو من الوسم والسمة وهي العلامة وكأنه علامة لمعناه والأول أصح لأنه يصغر على السمي ولو كان من السمة لكان يصغر على الوسيم كما يقال في الوعد وعيد ويقال في تصريفه سميت ولو كان من الوسم لقيل وسمت . قوله تعالى " الله " قال الخليل وجماعة',
                "tafsir_en": "Bismillah بِسْمِ اللَّـهِ is a verse of the Holy Qur'an\n\nThere is consensus of all the Muslims on the fact that Bismillah al-Rahman al-Rahim بِسْمِ اللَّـهِ الرَّ‌حْمَـٰنِ الرَّ‌حِيمِ is a verse of the Holy Qur'an, being a part of the Surah al-Naml سورة النمل (The Ant); and there is also an agreement on that this verse is written at the head of every Surah except the Surah al-Taubah سورة التوبہ . But there is a difference of opinion among the Mujtahids مجتہدین (the authentic scholars who are entitled to express an opinion in such matters) as to whether this verse is an integral part of the Surah al-Fatihah or of all the Surahs or not. According to the great Imam Abu Hanifah (رح) ، it is not an integral part of any Surah except al-Naml, rather it is in itself an independent verse of the Holy Qur'an which has been revealed for being placed at the beginning of every Surah in order to separate and distinguish one Surah from another.\n\nThe merits of Bismillah بسمِ اللہ\n\nIt was a custom in the Age of Ignorance (Jahiliyyah جاہلیہ ) before the advent of Islam that people began everything they did with the names of their idols or gods. It was to eradicate this practice that the first verse of the Holy Qur'an which the Archangel Jibra'il (علیہ السلام) brought down to the Holy Prophet صلى الله عليه وسلم commanded him to begin the Qur'an with the name of Allah اقْرَأْ بِاسْمِ رَبِّكَ \"Read with the name of your Lord.",
                "surah": {
                    "name": "الفاتحة",
                    "nAyah": 7,
                    "revelationOrder": 5,
                    "type": "meccan",
                    "start": 1,
                    "end": 7,
                    "id": surah,
                },
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
    surah = 1
    ayat = 1
    test_build_mock_data(surah, ayat)
    config = {
        "provider": "quranjsonloader",
        "uri": TEST_JSON_FILENAME,
        "lang": "en",
    }
    loader = QuranJSONLoaderComponent().build(**config)
    assert isinstance(loader, JSONLoader)
    data = loader.load()
    assert isinstance(data[0], Document)
    assert len(data) == 1
    assert data[0].metadata["source"] == f"Qu'ran [{surah}:{ayat}]"


def test_JSONLoaderComponent():
    config = {
        "provider": "jsonloader",
        "uri": TEST_JSON_FILENAME,
        "jq_schema": ".data[]",
        "content_key": ".tafsir_ar",
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