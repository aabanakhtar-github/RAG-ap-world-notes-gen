from langchain_core.runnables import RunnableSerializable 
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
from vectordb import InMemoryVectorDB 

def extract_titles(llm, db):
    """Extracts titles from documents using the LLM."""
    system_prompt = """Find the titles in the document. A common pattern for titles is to contain a piece of information, and a following paragraph. If the title is incomplete, or has no understandable meaning in relation to the text, do not output it. They usually have very broad relation to the following sentence as shown:\n

    example: Factors that Influenced Trade Among the factors that influenced trade include disease, new routes, and changing society. 
    answer: Factors that Influenced Trade
    why: The title is very disconnected and generalized compared to the text that follows it

    example: Food of the Mayans The Mayans ate various kinds of dishes. 
    answer: Food of the Mayans
    why: The title is very broad compared to the text that follows it

    example: Unit 1: Ancient Greek Civilization 
    answer: Unit 1: Ancient Greek Civilization
    why: It's an obvious title, since its alone

    The aforementioned texts are EXAMPLES and shouldn't be included in the answer unless they are in the document provided.

    Please output the text that you think are titles in the document. If it seems like a title, please don't hesitate to output it, as it can be vague.
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
    generate a summary sentence, and then add 2-3 summary bullet points that highlight important themes, continouties, and ideas. Format your answer like this: 

    EXAMPLE:
    [Summary Sentence]
    - [Summary Bullet Point 1]
    - [Summary Bullet Point 2]
    - [Summary Bullet Point 3] 

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


def concission(llm, notes):
    """Makes the llm abbreviate and concise-ify the notes"""
    pass