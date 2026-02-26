import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

DATA_PATH = r"D:\Python-Project\news-nexus\data\raw_pdfs"
DB_PATH = r"D:\Python-Project\news-nexus\data\chroma_db"

def ingest_documents():
    print(f"loading documents from {DATA_PATH}...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False)
    
    chunks = text_splitter.split_documents(raw_documents)
    print(f"split into {len(chunks)} chunks.")

    from langchain_ollama import OllamaEmbeddings
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    print("Initializing vector store(this may take a few minutes for large PDFs)...")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    BATCH_SIZE = 100
    total_chunks = len(chunks)
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        vector_db.add_documents(batch)
        print(f">processsing batch {i//BATCH_SIZE + 1} of {total_chunks//BATCH_SIZE + 1}({len(batch)}chunks)...")
    
    print("Vector store created sucessfully.")
    return len(raw_documents), len(chunks)
if __name__ == "__main__":
    os.makedirs(DB_PATH, exist_ok=True)
    
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"No PDF's found in {DATA_PATH} Please add files to RAG")
    else:
        ingest_documents()