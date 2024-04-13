"""
Module contains tests for QuranTextSplitter class
"""

from rag.splitters import QuranTextSplitter


def test_quran_text_splitter():
    chunk_size = 100
    chunk_overlap = 0
    splitter = QuranTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    text = "Они сказали: «Обратитесь в иудаизм или христианство, и вы последуете прямым путем». Скажи: «Нет, в религию Ибрахима (Авраама), который был ханифом и не был одним из многобожников». \[\[Иудеи и христиане призывали мусульман обратиться в их религию и заявляли, что только они следуют прямым путем, тогда как приверженцы всех остальных верований \- заблудшие. Аллах велел мусульманам ответить им убедительным образом и сказать: «Мы будем исповедовать религию Ибрахима, который поклонялся одному Аллаху и отворачивался от всего остального, который исповедовал единобожие и отрекался от язычества и многобожия. Только так мы сможем пройти прямым путем, ибо если мы отвернемся от его религии, то окажемся среди неверующих и заблудших».\]\]"

    l = len(text) // chunk_size + 1
    chunks = splitter.split_text(text)
    assert isinstance(chunks, list)
    # assert len(chunks) == l
