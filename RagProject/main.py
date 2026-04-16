# =============================================================
# RAG Pipeline - Main Query Interface
# Loads the vector DB, retrieves relevant chunks, and answers
# user questions using the LLM + prompt template.
# =============================================================


# --- Environment Setup ---
from dotenv import load_dotenv

load_dotenv()


# --- Imports ---
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import chatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# --- Step 1: Load Embedding Model ---
# Same embedding model used during database creation
embeddingModel = MistralAIEmbeddings()


# --- Step 2: Connect to Existing Vector Store ---
# Loads the persisted Chroma DB (created by create_database.py)
vectorStore = Chroma(persist_directory="chromaDb", embedding_function=embeddingModel)


# --- Step 3: Initialize the LLM ---
LLMmodel = chatMistralAI(model="mistral-small-latest")


# --- Step 4: Setup Retriever ---
# MMR (Maximal Marginal Relevance) balances relevance and diversity in results
retriever = vectorStore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.5},
)


# --- Step 5: Define Prompt Template ---
# Instructs the LLM to answer strictly from the retrieved context
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say : "I could not find the answer in the document."
"""),
    ("human", """ Context:{context} Question:{question} """)
])


# --- Step 6: Run the Query Loop ---
print("Rag system created")
print("Type 0 to exit the application")

while True:
    UserQuery = input("User:\n")

    if UserQuery == "0":
        break

    # Retrieve the most relevant chunks from the vector store
    docs = retriever.invoke(UserQuery)

    # Combine retrieved chunks into a single context string
    context = "\n\n".join([doc.page_content for doc in docs])

    # Format the prompt with context and user question
    finalPrompt = prompt.invoke({"context": context, "question": UserQuery})

    # Get the response from the LLM
    response = LLMmodel.invoke(finalPrompt)

    print("AI Assistant:\n", response.content)
