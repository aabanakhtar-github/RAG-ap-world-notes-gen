import getpass
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
from vectordb import InMemoryVectorDB 

if not os.environ.get("COHERE_API_KEY"):
  os.environ["COHERE_API_KEY"] = getpass.getpass("Enter API key for Cohere: ")

#vector_db = InMemoryVectorDB(path="../data/test.pdf")
#vector_db.create_database()

llm = ChatCohere(model="command-r-plus")

system = '''You are an agent that takes in a set of AP World History textbook pdfs, and are reponsible for responding to the user with
 the most relevant answer based on the provided context from the notes. Answer the following question
 '''
prompt = PromptTemplate(
  input_variables = ["question"],
  template=system + " {question}"
)


chain = (
  prompt 
  | llm 
  | StrOutputParser()
)

print(chain.invoke({"question", "what is your job ma'am?"}))
