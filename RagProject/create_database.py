# =============================================================
# RAG Pipeline - Database Creation
# Steps: Load PDF → Split into chunks → Embed → Store in DB
# =============================================================


# --- Environment Setup ---
from dotenv import load_dotenv
load_dotenv()


# --- Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# --- Step 1: Load PDF ---
PdfLoader = PyPDFLoader("exercise_science_research_paper.pdf")

pdfDocuments = PdfLoader.load()


# --- Step 2: Split into Chunks ---
RecursiveSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

chunks = RecursiveSplitter.split_documents(pdfDocuments)


# --- Step 3: Create Embeddings ---
embeddingModel = MistralAIEmbeddings()


# --- Step 4: Store Embeddings in Vector Database ---
vectorStore = Chroma.from_documents(documents=chunks, embedding=embeddingModel, persist_directory="chromaDb")
