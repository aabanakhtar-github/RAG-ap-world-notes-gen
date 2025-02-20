import getpass
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
from vectordb import InMemoryVectorDB 

if not os.environ.get("COHERE_API_KEY"):
  os.environ["COHERE_API_KEY"] = getpass.getpass("Enter API key for Cohere: ")

#vector_db = InMemoryVectorDB(path="../data/test.pdf")
#vector_db.create_database()

# temp = creativity
llm = ChatCohere(model="command-r-plus", temperature=0)
db = InMemoryVectorDB("../data/hehe.pdf")
db.create_database() 
#system = '''You are an agent that takes in a set of AP World History textbook pdfs, and are reponsible for responding to the user with
# the most relevant answer based on the provided context from the notes. Answer the following question
# '''

system = """Find the titles in the document. A common pattern for titles is to contain a piece of information, and a following paragraph. They usually have very broad relation to the following sentence as shown:\n
example: Factors that Influenced Trade Among the factors that influenced trade include disease, new routes, and changing society. 
answer: Factors that Influenced Trade, as it is a phrase before the sentence with no direct impact on the sentence
example: Food of the Mayans The Mayans ate various kinds of dishes. 
answer: Food of the Mayans
"""

retreiver = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
prompt = ChatPromptTemplate.from_template(system + "\n document: {doc}")

doc = ""
for i in db.documents():
  doc += i.page_content + "\n";


chain = (
  prompt 
  | llm 
  | StrOutputParser()
)
print(chain.invoke(prompt.invoke({"doc", doc})))