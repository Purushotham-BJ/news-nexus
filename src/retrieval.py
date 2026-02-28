import os
import sys
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DB_PATH=r"D:\Python-Project\news-nexus\data\chroma_db"
def retrieve_documents(query, k=4, keywords_filter=True):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

    results = vector_store.similarity_search_with_score(query, k=k+2)

    final_results = []

    for doc, score in results:
        final_results.append((doc, score))

    return final_results[:k]

if __name__ == "__main__":
    test_query = "What is the impact of GenAI on productivity?"
    retrieved_docs = retrieve_documents(test_query)
    print(f"\n---Top{len(retrieved_docs)}Results---")
    for i, (doc, score) in enumerate(retrieved_docs):
        print(f"\n [Results{i+1}] (Score: {score:.4f})")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Content Snippet : {doc.page_content[:200]}...")
