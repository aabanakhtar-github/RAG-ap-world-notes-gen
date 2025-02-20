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
system = '''You are an agent that takes in a set of AP World History textbook pdfs, and are reponsible for responding to the user with
 the most relevant answer based on the provided context from the notes. Answer the following question
 '''
query = "What are key factors in the expansion of trade?"

retreiver = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
docs = retreiver.invoke(query)
prompt = ChatPromptTemplate.from_template(system + "\n context: {context} \n question: {question}")
for i in docs:
  print (i.page_content)


chain = (
  prompt 
  | llm 
  | StrOutputParser()
)

