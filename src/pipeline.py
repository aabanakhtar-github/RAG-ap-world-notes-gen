import getpass
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import *
from langchain_cohere import ChatCohere
from vectordb import InMemoryVectorDB 
from chain_runnables import *

class NotesPipeline:
    '''Simple class that represents the RAG pipeline''' 
    def __init__(self, api_key, file):
        self.llm = ChatCohere(model="command-r-plus", cohere_api_key=api_key)
        self.db = InMemoryVectorDB(file, api_key)
        self.db.create_database() 

    def invoke(self) -> dict: 
        find_subchain = RunnableLambda(lambda _ : extract_titles(self.llm, self.db))
        clean_subchain = RunnableLambda(lambda input : format_titles(self.llm, input))
        generate_subchain = RunnableLambda(lambda titles : generate_notes(self.llm, self.db, titles))
        pipeline = find_subchain | clean_subchain | generate_subchain
        return pipeline.invoke({})