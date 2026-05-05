# =============================================================
# RAG Pipeline - Main Query Interface (Upgraded)
# Vertex AI Gemini LLM + Vertex AI Embeddings
# Hybrid Search: BM25 + Vector (MMR) + Cross-encoder Reranker
# LangSmith Tracing on every step
# =============================================================

# --- Environment Setup ---
from dotenv import load_dotenv
load_dotenv()

# Required .env variables:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=your_langsmith_api_key
#   LANGCHAIN_PROJECT=rag-pipeline
#   GOOGLE_APPLICATION_CREDENTIALS=path/to/your-service-account-key.json
#   GOOGLE_CLOUD_PROJECT=your-gcp-project-id

import os

# --- Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  # ← Vertex AI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langsmith import traceable


# =============================================================
# Step 1: Embedding Model (Vertex AI) - UPDATED
# =============================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location="us-central1",
    vertexai=True,
)


# =============================================================
# Step 2: Load Existing Vector Store
# =============================================================

vector_store = Chroma(
    persist_directory="chromaDb",
    embedding_function=embedding_model,
)


# =============================================================
# Step 3: LLM (Vertex AI Gemini)
# =============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location="us-central1",
    vertexai=True,
    temperature=0.2,
    max_output_tokens=1024,
    streaming=True,
)


# =============================================================
# Step 4: Hybrid Retriever (BM25 + Vector + Reranker)
# =============================================================

# --- 4a: Semantic retriever (MMR removes redundant chunks) ---
vector_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,          # final chunks returned
        "fetch_k": 20,   # candidates before MMR diversification
        "lambda_mult": 0.5,  # 0 = max diversity, 1 = max relevance
    },
)

# --- 4b: BM25 (keyword) retriever built from Chroma's stored docs ---
stored = vector_store.get(include=["documents", "metadatas"])
doc_objs = [
    Document(page_content=text, metadata=meta or {})
    for text, meta in zip(stored["documents"], stored["metadatas"])
]
bm25_retriever = BM25Retriever.from_documents(doc_objs)
bm25_retriever.k = 5

# --- 4c: Ensemble: BM25 (40%) + Vector (60%) ---
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
)

# --- 4d: Cross-encoder reranker (improves final ordering) ---
# Uses a small local model — no extra API key needed.
# Reranker scores each retrieved chunk against the query and re-orders them.
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)  # keep top 3 after reranking

retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever,
)


# =============================================================
# Step 5: Prompt Template
# =============================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise and helpful AI assistant.

Rules:
- Answer ONLY from the provided context.
- Be concise and structured in your response.
- If the answer is not in the context, say:
  "I could not find the answer in the provided documents."
- Do not hallucinate or add information not present in the context.
- Cite relevant details where useful.
"""),
    ("human", """Context:
{context}

Question: {question}

Answer:""")
])


# =============================================================
# Step 6: Format retrieved docs
# =============================================================

def format_docs(docs: list[Document]) -> str:
    """Join chunks with separators. Includes source metadata if available."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i} | {source}, p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# =============================================================
# Step 7: Build RAG Chain
# =============================================================

rag_chain = (
    {
        "context":  retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# =============================================================
# Step 8: Traceable Query Function
# =============================================================

@traceable(
    name="rag_query",
    # LangSmith will log: input, retrieved docs, prompt, LLM response, latency
    metadata={"retriever": "hybrid-bm25-mmr", "llm": "gemini-2.5-flash"},
)
def run_query(question: str) -> str:
    """
    Run a single query through the RAG pipeline.
    Decorated with @traceable so every call is logged in LangSmith.
    """
    result = []
    print("AI Assistant:")
    for chunk in rag_chain.stream(question):
        # Print incrementally for token-like terminal visibility.
        for ch in chunk:
            print(ch, end="", flush=True)
        result.append(chunk)
    print()  # newline after streaming finishes
    return "".join(result)


# =============================================================
# Step 9: Query Loop
# =============================================================

    if __name__ == "__main__":
        print("=" * 60)
        print("RAG System Ready")
        print("Retrieval: Hybrid (BM25 + Vector MMR) + Cross-Encoder Rerank")
        print("LLM      : Vertex AI Gemini 2.5 Flash")
        print("Tracing  : LangSmith (check your project dashboard)")
        print("Type '0' to exit")
        print("=" * 60)

        while True:
            user_query = input("\nUser:\n").strip()
            ()

            if not user_query:
                continue

            if user_query == "0":
                print("Goodbye!")
                break

            run_query(user_query)
