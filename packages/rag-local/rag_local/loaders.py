# -*- coding: utf-8 -*-
# loaders.py
"""
Module contains adapters classes for text loaders by different providers.
"""

# pylint: disable=no-name-in-module
from typing import Dict, Sequence, Union
from bs4 import BeautifulSoup as Soup
from logger import logger
from langchain_community.document_loaders import WebBaseLoader, TextLoader, JSONLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from rag_local.component import raise_not_implemented

# from langchain_community.document_loaders import json_loader


def build_loader(config: Dict):
    """Build the loader based on the config dict

    Args:
        config (Dict): dictionary with the following keys:
            provider (str): WebBaseLoader | JsonLoader | TextLoader | RecursiveUrlLoader
            uri (str) : uri of the document to load
            jq_schema (str) : jq schema [https://python.langchain.com/docs/modules/data_connection/document_loaders/json]
            encoding (str): utf-8 | ascii

    Returns:
        loader

    Raises:
        Exception: NotImplementedError if unknown provider
    """

    providers = {
        "webbaseloader": WebBaseLoaderComponent().build,
        "jsonloader": JSONLoaderComponent().build,
        "quranjsonloader": QuranJSONLoaderComponent().build,
        "textloader": TextLoaderComponent().build,
        "recursiveloader": RecursiveUrlLoaderComponent().build,
    }
    loader = providers.get(config["provider"].lower(), raise_not_implemented)(**config)
    return loader


class WebBaseLoaderComponent:
    display_name = "WebBaseLoaderComponent"
    description = "Web Loader Component"
    documentation = ""

    def build(self, uri: Union[str, Sequence[str]] = "", **kwargs) -> WebBaseLoader:
        return WebBaseLoader(uri)


class RecursiveUrlLoaderComponent:
    display_name = "RecursiveUrlLoaderComponent"
    description = "Recursive URL Loader Component"
    documentation = ""

    def build(self, uri: str, max_depth: int = 2, **kwargs) -> RecursiveUrlLoader:
        return RecursiveUrlLoader(
            url=uri, max_depth=max_depth, extractor=lambda x: Soup(x, "html.parser")  # type: ignore
        )


class TextLoaderComponent:
    display_name = "TextLoaderComponent"
    description = "Text file Loader"
    documentation = ""

    def build(
        self, uri: Union[str, Sequence[str]] = "", encoding: str = "utf-8", **kwargs
    ) -> TextLoader:
        return TextLoader(file_path=uri, encoding=encoding)


class JSONLoaderComponent:
    display_name = "JsonLoaderComponent"
    description = "Json file Loader"
    documentation = ""

    def build(
        self,
        uri: Union[str, Sequence[str]] = ".",
        jq_schema: str = "",
        content_key: str = "",
        **kwargs,
    ) -> JSONLoader:

        return JSONLoader(
            file_path=uri,
            jq_schema=jq_schema,
            content_key=content_key,
            is_content_key_jq_parsable=True,
        )


class QuranJSONLoaderComponent:
    display_name = "JsonLoaderComponent"
    description = "Json file Loader"
    documentation = ""

    def build(
        self,
        uri: Union[str, Sequence[str]] = ".",
        lang: str = "en",
        **kwargs,
    ) -> JSONLoader:
        # Field of json-record to use in parser
        content_cases = {
            "ar": ".tafsir_ar",
            "en": ".tafsir_en",
            "ru": ".tafsir_ru",
        }

        if lang not in content_cases:
            logger.error(f"Language {lang} is not supported. Using 'en'.")
            lang = "en"
        jq_schema = ".data[]"
        content_key = content_cases[lang]

        return JSONLoader(
            file_path=uri,
            jq_schema=jq_schema,
            content_key=content_key,
            is_content_key_jq_parsable=True,
            metadata_func=self.metadata_func,
        )

    def metadata_func(self, record: dict, metadata: dict) -> dict:
        """Extract and transform metadata from record of json file."""
        ayat_id = record["id"]
        surah_id = record["surah"]["id"]
        if "source" in metadata:
            metadata["source"] = f"Qu'ran [{ayat_id}:{surah_id}]"
            # TODO: Add description of used tafsir. Example:
            # Name    : Tafseer Al Qurtubi
            # Author  : Qurtubi
            # Language: Arabic
            # Source  : Quran.com
        return metadata


# Qu'ran json format example
# { data: [
# {
#     "id": 3770,
#     "arabic": "ٱلْيَوْمَ نَخْتِمُ عَلَىٰٓ أَفْوَٰهِهِمْ وَتُكَلِّمُنَآ أَيْدِيهِمْ وَتَشْهَدُ أَرْجُلُهُم بِمَا كَانُوا۟ يَكْسِبُونَ",
#     "tafsir_ar": "( نختم على أفواههم وتكلمنا أيديهم وتشهد أرجلهم بما كانوا يكسبون ) هذا حين ينكر الكفار كفرهم وتكذيبهم الرسل ، فيختم على أفواههم وتشهد عليهم جوارحهم .أخبرنا عبد الواحد بن أحمد المليحي ، أخبرنا أبو الحسن محمد بن عمرو بن حفصويه السرخسي ، سنة خمس وثلاثين وثلاثمائة ، أخبرنا أبو يزيد حاتم بن محبوب ، أخبرنا عبد الجبار بن العلاء ، أخبرنا سفيان عن سهيل بن أبي صالح عن أبيه عن أبي هريرة قال : سأل الناس رسول الله \\- صلى الله عليه وسلم \\- فقالوا : يا رسول الله هل نرى ربنا يوم القيامة ؟ فقال : \" هل تضارون في رؤية الشمس في الظهيرة ليست في سحاب \" ؟ قالوا : لا يا رسول الله ، قال : \" فهل تضارون في رؤية القمر ليلة البدر ليس في سحابة \" ؟ قالوا : لا ، قال : \" فوالذي نفسي بيده لا تضارون في رؤية ربكم كما لا تضارون في رؤية أحدهما \" ، قال : \" فيلقى العبد فيقول : أي عبدي ألم أكرمك ؟ ألم أسودك ألم أزوجك ألم أسخر لك الخيل والإبل وأذرك تترأس وتتربع ؟ قال : بلى يا رب ، قال : فظننت أنك ملاقي ؟ قال : لا ، قال : فاليوم أنساك كما نسيتني . قال : فيلقى الثاني فيقول : ألم أكرمك ، ألم أسودك ، ألم أزوجك ، ألم أسخر لك الخيل والإبل وأتركك تترأس وتتربع ؟ \\- وقال غيره عن سفيان : ترأس وتربع في الموضعين \\- قال : فيقول : بلى يا رب ، فيقول : ظننت أنك ملاقي ؟ فيقول : لا يا رب قال : فاليوم أنساك كما نسيتني . ثم يلقى الثالث فيقول ؟ ما أنت ؟ فيقول : أنا عبدك آمنت بك وبنبيك وبكتابك وصليت وصمت وتصدقت ويثني بخير ما استطاع قال : فيقال له : ألم نبعث عليك شاهدنا ؟ قال : فيتفكر في نفسه من الذي يشهد علي فيختم على فيه ، ويقال لفخذه : انطقي قال : فتنطق فخذه ولحمه وعظامه بما كان يعمل ، وذلك المنافق ؛ وذلك ليعذر من نفسه وذلك الذي سخط الله عليه \" .أخبرنا أبو سعيد عبد الله بن أحمد الطاهري ، أخبرنا جدي أبو سهل عبد الصمد بن عبد الرحمن البزاز ، أخبرنا محمد بن زكريا العذافري ، أخبرنا إسحاق بن إبراهيم الدبري ، أخبرنا عبد الرزاق ، أخبرنا معمر ، عن بهز بن حكيم بن معاوية ، عن أبيه عن جده عن النبي \\- صلى الله عليه وسلم \\- قال : \" إنكم تدعون فيفدم على أفواهكم بالفدام فأول ما يسأل عن أحدكم فخذه وكفه \"أخبرنا إسماعيل بن عبد القاهر ، أخبرنا عبد الغافر بن محمد ، أخبرنا محمد بن عيسى الجلودي ، أخبرنا إبراهيم بن محمد بن سفيان ، أخبرنا مسلم بن الحجاج ، أخبرنا أبو بكر بن أبي النضر ، حدثني هاشم بن القاسم ، أخبرنا عبد الله الأشجعي ، عن سفيان الثوري ، عن عبيد المكتب ، عن فضيل ، عن الشعبي ، عن أنس بن مالك قال : كنا عند رسول الله \\- صلى الله عليه وسلم \\- فضحك فقال : \" هل تدرون مم أضحك \" ؟ قال : قلنا الله ورسوله أعلم ، قال : \" من مخاطبة العبد ربه \" يقول : يا رب ألم تجرني من الظلم ؟ قال : فيقول : بلى ، قال : فيقول : فإني لا أجير على نفسي إلا شاهدا مني ، قال : فيقول : كفى بنفسك اليوم عليك شهيدا وبالكرام الكاتبين شهودا ، قال : فيختم على فيه ، فيقال لأركانه : انطقي قال : فتنطق بأعماله ، قال : ثم يخلى بينه وبين الكلام ، فيقول : بعدا لكن وسحقا فعنكن كنت أناضل \"",
#     "tafsir_en": "In verse 65, it was said: الْيَوْمَ نَخْتِمُ عَلَىٰ أَفْوَاهِهِمْ (Today We will set a seal on their mouths). On the day of Resurrection, when comes the time to account for deeds, everyone will be free to offer any excuse one has. But, Mushriks, the practitioners of shirk, those who associate partners in the pristine divinity of Allah Ta’ ala, will declare on oaths that they never had anything to do with shirk and kufr: وَاللَّـهِ رَ‌بِّنَا مَا كُنَّا مُشْرِ‌كِينَ (By Allah, our Lord, we ascribed no partners to Allah - A1-An\\` am, 6:23).\n\nAnd some of them will also say that they were free of whatever the angels had written down in their book of deeds. At that time, Allah Ta’ ala will put a seal on their mouths, so that they would not speak. Then, He will give power of speech to their own body parts, the hands and the feet, who will testify to all their deeds as court witnesses against them. As for the present verse, it mentions the speaking of hands and feet only. In another verse, mentioned there is the speaking of one's ear, eye and skin: شَهِدَ عَلَيْهِمْ سَمْعُهُمْ وَأَبْصَارُ‌هُمْ وَجُلُودُهُم ' (their ears and their eyes and their skins will testify against them - 41:20). As for what has been said at one place: تَشْهَدُ عَلَيْهِمْ أَلْسِنَتُهُمْ (and their tongues will testify against them - An-Nur, 24:24), it is not contrary to 'putting a seal on their mouths' because putting a seal means that they will be unable to say anything out of their own volition. Their tongue will speak counter to their personal choice and will testify to the truth.\n\nAs for the question how these parts of the body would acquire power of speech, the Qur'an has already answered that by saying: أَنطَقَنَا اللَّـهُ الَّذِي أَنطَقَ كُلَّ شَيْءٍ (Why did you testify against us? - 41:21) that is, these parts of the body will say that Allah, who has given power of speech to all things endowed with the ability to speak, has also enabled us to speak.",
#     "tafsir_ru": "Сегодня Мы запечатаем их уста. Их руки будут говорить с Нами, а их ноги будут свидетельствовать о том, что они приобретали. \\[\\[Всевышний описал ужасное состояние неверующих в Последней жизни. Они не смогут отрицать своего неверия и совершенных ими грехов. Каждая часть тела будет признаваться в совершенных ими грехах, ибо Всевышний Аллах может заставить заговорить всякую вещь.\\]\\]",
#     "surah": {
#         "name": "يس",
#         "nAyah": 83,
#         "revelationOrder": 41,
#         "type": "meccan",
#         "start": 3706,
#         "end": 3788,
#         "id": 36
#     }
# }
# ]
# }
