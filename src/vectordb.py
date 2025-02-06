from langchain_community.document_loaders import PyMuPDFLoader 
# create the loader 
loader = PyMuPDFLoader("../data/test.pdf")
pages = loader.load()
print(f"{pages[0].metadata}")
print("contents")
print(f"{pages[0].page_content}")