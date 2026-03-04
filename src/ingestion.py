import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv() # Loads your OPENAI_API_KEY from .env

def build_vector_db():
    # 1. Load PDFs
    loader = PyPDFDirectoryLoader("data/guidelines/")
    docs = loader.load()
    
    # 2. Split text (Architectural choice: small chunks for precision)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # 3. Create Embeddings & Store in ChromaDB
    print(f"Ingesting {len(chunks)} chunks into PulsePoint Vector Store...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory="vector_db/"
    )
    print("Ingestion Complete!")

if __name__ == "__main__":
    build_vector_db()