from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

text = ["My name is Rachit", "I am a student", "I am a human"]

embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
)

vector = embeddings.embed_documents(text)
print(vector)