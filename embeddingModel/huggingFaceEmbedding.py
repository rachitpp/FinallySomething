from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts =["My name is Rachit","I am a student","I am a human"]

vector = embeddings.embed_documents(texts)
print(vector.content)