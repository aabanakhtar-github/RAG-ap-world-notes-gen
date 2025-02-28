from langchain_core.runnables import RunnableSerializable 
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
from vectordb import InMemoryVectorDB 
import copy

def extract_titles(llm, db):
    """Extracts titles from documents using the LLM."""
    system_prompt = """Find the titles in the document. A title is characterized by the following properties:
        Key Concepts or Themes: Titles reflect the main idea or theme (e.g., "trade").
        Relevance of Keywords: Focus on significant terms that represent the core of the text.
        Structure of the Text: Titles often come from the introductory sentences or headings.
        Conciseness: Keep titles brief and focused on key aspects.
        Contextual Clues: Words like "Influenced," "Factors," or "Conditions" can signal a title.
        Presence of a Subject: Titles typically have a subject (e.g., "Trade") and an action or condition (e.g., "Influenced").
        Avoidance of Details: Titles avoid excessive detail unless essential.
        Language Patterns: Titles often use noun phrases or base form verbs.

    example: Factors that Influenced Trade Among the factors that influenced trade include disease, new routes, and changing society. Traders also faced dangerous conditions when...
    answer: Factors that Influenced Trade
    why: The title is very disconnected and generalized compared to the text that follows it, and "Factors that influenced Trade" is a sentence fragment

    example: Food of the Mayans The Mayans ate various kinds of dishes. Among these dishes were beans, maize, and potatoes...
    answer: Food of the Mayans
    why: The title is a sentence fragment proceeding a paragraph. It is relatively broad compared to the following information.

    example: Unit 1: Ancient Greek Civilization 
    answer: Unit 1: Ancient Greek Civilization
    why: It's an obvious title, since its alone

    The aforementioned texts are EXAMPLES and shouldn't be included in the answer unless they are in the document provided.

    Please output the text that you think are titles in the document. Explain your reasoning after each title. 
    """

    prompt = ChatPromptTemplate.from_template(system_prompt + "\nDocuments:\n{doc}")
    chain = prompt | llm | StrOutputParser()
    doc = "\n".join(f"Source {i}:\n{i.page_content}" for i in db.documents())

    return chain.invoke({"doc": doc})


def format_titles(llm, input_text: str) -> list[str]:
    """Cleans and formats extracted titles."""
    system_prompt = """You will be provided the answer of another agent, which contains titles from a text. Please do the following:
    - Remove any response text (example: "Here is what I found")
    - Remove any annotations (example: "(repeated title)")
    - Filter the Titles that make no semantic sense
    - Remove any form of list annotation (numbering, dashes, bullets, etc.)
    - Remove any duplicates 
    - Order them in a list, with no bullets, dashes, or anything. Separate each title with a newline

    Your answer should like this: 
    title a
    title b
    title c
    """

    prompt = ChatPromptTemplate.from_template(system_prompt + "\nData:\n{data}")
    chain = prompt | llm | StrOutputParser()
    value = chain.invoke({"data": input_text})
    return value.splitlines()

def generate_notes(llm, db, titles: list[str]):
    """Generates the actual two column notes using the titles from the previous chain"""
    system_prompt = """You will be provided a title and a list of documents. Based on the title, and the documents,
    generate a summary sentence, and then add 2-3 summary bullet points that highlight important themes, continouties, and ideas. Strictly Format your answer like this, without adding any response fluff
    Specifically, do not add headers like SUMMARY SENTENCE, or SUMMARY BULLET POINTS: 

    EXAMPLE FORMAT:
    Summary Sentence
    - Summary Bullet Point 1
    - Summary Bullet Point 2
    - Summary Bullet Point 3 

    Please make them as concise and short as possible, while maintaining their semantic value. Use symbols, abbreviations, and acronyms. Simplify words by removing their vowels if they make sense ("people" -> "ppl")
    """
    notes = dict()
    prompt = ChatPromptTemplate.from_template(system_prompt + "Documents:\n{documents}\nTitle:{title}\n")
    chain = prompt | llm | StrOutputParser() 
    retriever = db.as_retriever()
    for title in titles: 
        docs = retriever.invoke(title)
        docs_context = "\n".join(f"Source {i}:\n{i.page_content}" for i in docs)
        notes[title] = chain.invoke({"documents" : docs_context, "title" : title})

    return notes