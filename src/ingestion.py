import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv() # Loads your OPENAI_API_KEY from .env

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
def build_vector_db():
    # 1. Load PDFs
    loader = PyPDFDirectoryLoader("data/guidelines/")
    raw_docs = loader.load()
    
    # 2. Add Contextual Headers to every page
    # This ensures the year and guideline title are in EVERY chunk for both BM25 and Vector search.
    docs = []
    for d in raw_docs:
        filename = os.path.basename(d.metadata.get('source', 'Unknown'))
        doc_title = filename.replace(".pdf", "").replace("-", " ")
        d.page_content = f"--- DOCUMENT: {doc_title} ---\n{d.page_content}"
        docs.append(d)
    
    # 3. Define Splitters (Recursive refined for medical text)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)

    # 3. Setup Persistent Storage for Parents
    # We use a LocalFileStore to persist parent documents on disk
    fs = LocalFileStore("./parent_store")
    store = create_kv_docstore(fs)
    
    # 4. Initialize Vector Store
    vectorstore = Chroma(
        collection_name="stroke_rag_parents", 
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory="vector_db_parent/"
    )

    # 5. Initialize ParentDocumentRetriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 6. Add documents to the retriever in batches
    # We batch the input docs because ChromaDB (especially with Child/Parent split) 
    # can exceed its max batch size of ~5461.
    print(f"Indexing {len(docs)} guideline pages with Parent Document Retrieval...")
    batch_size = 50
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(docs)-1)//batch_size + 1}...")
        retriever.add_documents(batch)
    
    print("Parent Document Ingestion Complete!")

if __name__ == "__main__":
    build_vector_db()