# =============================================================
# RAG Pipeline - Main Query Interface (Hybrid Search Version)
# Loads the vector DB, retrieves relevant chunks via hybrid
# search (BM25 + Vector), and answers user questions using
# the LLM + prompt template.
# =============================================================


# --- Environment Setup ---
from dotenv import load_dotenv

load_dotenv()


# --- Imports ---
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever



# --- Step 1: Load Embedding Model ---
embeddingModel = MistralAIEmbeddings()


# --- Step 2: Connect to Existing Vector Store ---
vectorStore = Chroma(persist_directory="chromaDb", embedding_function=embeddingModel)


# --- Step 3: Initialize the LLM ---
LLMmodel = ChatMistralAI(model="mistral-small-latest")


# --- Step 4: Setup Hybrid Retriever (BM25 + Vector) ---

# Semantic (vector) retriever using MMR
vector_retriever = vectorStore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)

# Keyword (BM25) retriever built from the same chunks stored in Chroma
stored = vectorStore.get(include=["documents", "metadatas"])
doc_objs = [
    Document(page_content=text, metadata=meta or {})
    for text, meta in zip(stored["documents"], stored["metadatas"])
]

bm25_retriever = BM25Retriever.from_documents(doc_objs)
bm25_retriever.k = 5

# Combine both: weights = [BM25, Vector]
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
)


# --- Step 5: Define Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say : "I could not find the answer in the document."
"""),
    ("human", """ Context:{context} Question:{question} """)
])


# --- Step 6: Format retrieved docs into a single context string ---
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# --- Step 7: Build the RAG Chain using Runnables ---
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | LLMmodel
    | StrOutputParser()
)


# --- Step 8: Run the Query Loop ---
print("Rag system created (Hybrid Search: BM25 + Vector)")
print("Type 0 to exit the application")

while True:
    UserQuery = input("User:\n").strip()

    if UserQuery == "0":
        break
    

    print("AI Assistant:")
    for chunk in rag_chain.stream(UserQuery):
        print(chunk, end="", flush=True)
    print()