from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.documents import Document


docs = [Document(page_content="Hello this is rachit",metadata={"source":"notes1.txt"}),
        Document(page_content="Hello this is rachit",metadata={"source":"notes2.txt"}),
        Document(page_content="Hello this is rachit",metadata={"source":"notes3.txt"}),
  ]


embeddingModel = MistralAIEmbeddings()

vectorStore = Chroma.from_documents(documents=docs,embedding=embeddingModel,persist_directory="chroma_db")

result = vectorStore.similarity_search("What is used for data analysis?",k=2)


for r in result:
    print(r)

print(vectorStore)