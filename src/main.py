import getpass
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_cohere import ChatCohere 

if not os.environ.get("COHERE_API_KEY"):
  os.environ["COHERE_API_KEY"] = getpass.getpass("Enter API key for Cohere: ")
  
model = ChatCohere(model = "command-r-plus")
history = []

model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)