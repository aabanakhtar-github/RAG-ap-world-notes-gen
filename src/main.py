import getpass
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_cohere import ChatCohere
from vectordb import InMemoryVectorDB 

if not os.environ.get("COHERE_API_KEY"):
  os.environ["COHERE_API_KEY"] = getpass.getpass("Enter API key for Cohere: ")
  
InMemoryVectorDB("../data/random.pdf").createDB()