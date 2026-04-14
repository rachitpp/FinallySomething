from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    dimensions = 64
)
texts = ["Hello this is rachit for you",'Hello your name is Youtube',"And you all are decent"] 


# this is the vector representation of the query , here we create a vector for the query using the embedding
# here we use embed.query to create vector for a single one line query
vector = embeddings.embed_query("You are going to learn Gen AI")

# creating vector for multiple lines of query using embed.documents
vector1 = embeddings.embed_documents(texts)
print(vector)
print(vector1)