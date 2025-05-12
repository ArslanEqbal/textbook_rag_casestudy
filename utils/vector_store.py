# utils/vector_store.py
import json
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def ontology_to_documents(ontology_data):
    docs = []
    for item in ontology_data:
        concept = item.get("concept", "Unknown Concept")
        definition = item.get("definition", "No definition available.")
        related_terms = item.get("related_terms", [])

        content = f"Concept: {concept}\nDefinition: {definition}\n\nRelated_Terms: {related_terms}"
        docs.append(Document(page_content=content, metadata={"concept": concept}))
    return docs

def create_vector_db(docs, persist_dir="chroma_book_ontology_db"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=persist_dir)
    db.persist()
    return db

def load_vector_db(persist_dir="chroma_book_ontology_db"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
