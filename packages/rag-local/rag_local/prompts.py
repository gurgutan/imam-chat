# -*- coding: utf-8 -*-
# llms.py
"""
Module contains adapters for prompt templates
"""

from langchain_core.prompts import ChatPromptTemplate


# Prompt for muslim imam chat
class MuslimImamPrompt:
    description = "Prompt for answer query with context ин muslim imam"
    documentation = ""
    template = """Act as a Muslim imam who gives me guidance and advice on how to deal with life problems. Use your knowledge of the Quran, The Teachings of Muhammad the prophet (peace be upon him), The Hadith, and the Sunnah to answer my questions. Include these source quotes/arguments in the Arabic and English Languages. 
    Answer the question based only on the following context:
    
        {context}

        Question: {question}
        
        Answer: """

    def build(self, **kwargs) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(self.template)


# Standart QA with context prompt
class QuestionAnswerPrompt:
    display_name = "AnswerQueryWithContextPrompt"
    description = "Prompt for answer query with come context"
    documentation = ""
    template = """Answer the question based only on the following context:
    
        {context}

        Question: {question}
        
        Answer: """

    def build(self, **kwargs) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(self.template)


# Пример промпта с цепочкой рассуждений
class QuestionAnswerCoTPrompt:
    display_name = "AnswerQueryWithContextPrompt"
    description = "Prompt for answer query with come context"
    documentation = ""
    step_delimiter = "####"
    template = f"""Analize the context and answer the user's query by following these steps. Use '{step_delimiter}' to delineate each step.
        Step 1:{step_delimiter} Determine if the query pertrains to Holy Quran or Islam.
        
        Step 2:{step_delimiter} Identify if the query is related to a specific surahs or ayahs.
        
        Step 3:{step_delimiter} Answer the query in the given context with including citation on Arabic or Russian language.
        
        Step 4:{step_delimiter} Correct any misconceptions, referencing only the context, and respond in a courteous manner.
    
        Context:
        
        {{context}}

        Question: {{question}}
        
        Answer: """

    def build(self, **kwargs) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(self.template)


# Пример промпта для ИИ-имама
# Ты ИИ ассистент, обладающий обширными знаниями в области Ислама и священного Корана.
# Отвечай как мусульманский имам, помогающий в толковании действий, поступков, мыслей, происшествий, наблюдаемых явлений с точки зрения священного Корана.
# Используй свои знания Корана, Учения пророка Мухаммеда (мир ему), хадисов и Сунны, чтобы отвечать на вопросы.
# Включай цитаты/аргументы из источников на арабском и английском языках.
