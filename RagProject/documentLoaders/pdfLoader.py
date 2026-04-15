from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


pdf_path = Path(__file__).resolve().parent / "exercise_science_research_paper.pdf"
data_loader = PyPDFLoader(str(pdf_path))
pdf_documents = data_loader.load()

print(pdf_documents)
print(len(pdf_documents))

# The length of document depends on the number of pages in the pdf file . 1 page = 1 content and metadata pair 
