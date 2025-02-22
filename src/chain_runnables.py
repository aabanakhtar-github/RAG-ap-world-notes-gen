from langchain_core.runnables import RunnableSerializable 
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
from vectordb import InMemoryVectorDB 

class TitleFinder:
    def __init__(self, llm, db):
        self.llm = llm 
        self.db = db
        self.system = """Find the titles in the document. A common pattern for titles is to contain a piece of information, and a following paragraph. If the title is incomplete, or has no understandable meaning in relation to the text, do not output it. They usually have very broad relation to the following sentence as shown:\n

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

        Please output the titles you find in the document.  If it seems like a title, please don't hesitate to output it, as it can be vague.
        """
        self.prompt = ChatPromptTemplate.from_template(self.system + " \nDocuments:\n {doc}")
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self) -> str:
       return self.chain.invoke({"doc" : self._create_documents()}) 

    def _create_documents(self) -> str:
        retreiver = MultiQueryRetriever.from_llm(retriever=self.db.as_retriever(), llm=self.llm)
        doc = ""
        for i in self.db.documents():
            doc += f"Source {i}:\n" + i.page_content + "\n";
        return doc

class FilterFormat:
    def invoke():
        pass 