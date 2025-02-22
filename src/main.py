import getpass
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
from vectordb import InMemoryVectorDB 
from chain_runnables import TitleFinder 

if not os.environ.get("COHERE_API_KEY"):
  os.environ["COHERE_API_KEY"] = getpass.getpass("Enter API key for Cohere: ")

llm = ChatCohere(model="command-r-plus", temperature=0.30)
db = InMemoryVectorDB("../data/hehe.pdf")
db.create_database() 
chain = TitleFinder(llm, db)

res = chain.invoke()

print(res)