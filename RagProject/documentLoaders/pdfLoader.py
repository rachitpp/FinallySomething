from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

pdf_path = Path(__file__).resolve().parent / "exercise_science_research_paper.pdf"
data_loader = PyPDFLoader(str(pdf_path))
pdf_documents = data_loader.load()

# print(pdf_documents)
# print(len(pdf_documents))

# The length of document depends on the number of pages in the pdf file . 1 page = 1 content and metadata pair 


token_splitter = TokenTextSplitter(chunk_size=1000,chunk_overlap=20)

chunks = token_splitter.split_documents(pdf_documents)

print(len(chunks))

print(chunks[0].page_content)