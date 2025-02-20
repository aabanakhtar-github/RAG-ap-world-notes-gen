from langchain_community.document_loaders import UnstructuredPDFLoader 
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import CharacterTextSplitter

class InMemoryVectorDB: 
    '''A class to represent the note taker's files in pdf form'''

    def __init__(self, path):
        # Use unstructured to split the pdf
        print("loading files....")
        self.loader = UnstructuredPDFLoader(path, mode="elements", strategy="hi_res")
        self.pages = self.loader.load() 
        # use cohere to embed and create store in memory
        print("creating vector store....")
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
        )
        self.store = InMemoryVectorStore(embeddings)
        print("created vector store!")

    def print_documents(self):
        [print(doc.metadata) for doc in self.pages]

    def documents(self):
        return self.pages

    def as_retriever(self): 
        return self.store.as_retriever(search_kwargs={"k":5})

    def create_database(self):
        # chunkviz.up.railway.app
        splitter = CharacterTextSplitter(
            separator=".",
            chunk_size=475,
            chunk_overlap=50,
            length_function=len
        )
        # make more readable chunks for the llm
        self.pages = splitter.split_documents(self.pages)
        self.store.add_documents(self.pages) 
        print("created database successfully!")