from langchain_community.document_loaders import UnstructuredPDFLoader 
# create the loader 
path = "../data/test.pdf"
loader = UnstructuredPDFLoader(path, mode="elements") 
pages = loader.load()
print(f"{pages[3].metadata}")
print("contents")
print(f"{pages[3].page_content}")
