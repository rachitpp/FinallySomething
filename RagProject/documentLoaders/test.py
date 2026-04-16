from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from pathlib import Path
data = TextLoader(str(Path(__file__).with_name("notes1.txt")))

# Creating a text splitter

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
    )
documents =data.load()
# Any kind of document that you create will have two things , its metadata and its content
# Here data.load returns a list of documents, and each document is a tuple of (content,metadata),

# printing the first document both content and metadata
# print(documents[0])

# # printing only the content of the first document
# print(documents[0].page_content)
# # printing only the metadata of the first document
# print(documents[0].metadata)

# # printing length of the documents list
# print(len(documents))




chunks = text_splitter.split_documents(documents)
print(len(chunks))