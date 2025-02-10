from langchain_community.document_loaders import UnstructuredPDFLoader 
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import CharacterTextSplitter

class InMemoryVectorDB: 
    '''A class to represent the note taker's files in pdf form'''

    def __init__(self, path):
        #self.loader = UnstructuredPDFLoader(path, mode = "single")
        print("loading files....")
        self.pages = Document(
            page_content="Hello, world! PME is oso coool!",
            metadata={"source": "https://example.com"}
        )
        print("creating vector store....")
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
        )
        self.store = InMemoryVectorStore(embeddings)

    def createDB(self):
        # chunkviz.up.railway.app
        splitter = CharacterTextSplitter(
            separator = ".",
            chunk_size= 475,
            chunk_overlap = 50,
            length_function = len
        )
        self.pages = splitter.split_documents([self.pages])
        self.store.add_documents(self.pages) 
        print(self.store.as_retriever().invoke("PME"))