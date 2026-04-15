from langchain_community.document_loaders import TextLoader

data = TextLoader("notes.txt")

documents =data.load()
# Any kind of document that you create will have two things , its metadata and its content
# Here data.load returns a list of documents, and each document is a tuple of (content,metadata),

# printing the first document both content and metadata
print(documents[0])

# printing only the content of the first document
print(documents[0].page_content)
# printing only the metadata of the first document
print(documents[0].metadata)

# printing length of the documents list
print(len(documents))