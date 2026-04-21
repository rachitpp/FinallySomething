# =============================================================
# RAG Pipeline - Main Query Interface (Hybrid Search + Memory)
# Loads the vector DB, retrieves relevant chunks via hybrid
# search (BM25 + Vector), and answers user questions using
# the LLM + prompt template. Supports conversational follow-ups.
# =============================================================


# --- Environment Setup ---
from dotenv import load_dotenv

load_dotenv()


# --- Imports ---
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# --- Step 1: Load Embedding Model ---
embeddingModel = MistralAIEmbeddings()


# --- Step 2: Connect to Existing Vector Store ---
vectorStore = Chroma(persist_directory="../RagProject/chromaDb", embedding_function=embeddingModel)


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


# --- Step 5: Contextualize prompt (rewrites follow-up questions) ---
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest user question, "
               "rewrite it as a fully standalone question. Do NOT answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    LLMmodel, retriever, contextualize_q_prompt
)


# --- Step 6: QA prompt ---
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say : "I could not find the answer in the document."

{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# --- Step 7: Build the RAG Chain ---
qa_chain = create_stuff_documents_chain(LLMmodel, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)


# --- Step 8: Run the Query Loop ---
print("Rag system created (Hybrid Search: BM25 + Vector + Memory)")
print("Type 0 to exit the application")

chat_history = []

while True:
    UserQuery = input("User:\n").strip()

    if UserQuery == "0":
        break

    result = rag_chain.invoke({"input": UserQuery, "chat_history": chat_history})
    answer = result["answer"]

    print("AI Assistant:")
    print(answer)
    print()

    chat_history.append(HumanMessage(content=UserQuery))
    chat_history.append(AIMessage(content=answer))