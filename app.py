# app.py
import streamlit as st
import os
from utils.ontology_extractor import extract_ontology
from utils.vector_store import ontology_to_documents, create_vector_db, load_vector_db
from utils.qa_chain import create_qa_chain
import requests

st.set_page_config(page_title="📘 RAG Ontology QA", layout="wide")
st.title("📘 Dynamic Q&A on Textbooks")


# Sidebar
st.sidebar.header("📄 Upload or Enter URL")

# Option to upload file or paste URL
input_option = st.sidebar.radio("Choose input method", ["Upload .txt File", "Paste URL"])

uploaded_text = None
uploaded_file = None

if input_option == "Upload .txt File":
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        uploaded_text = uploaded_file.read().decode("utf-8")

elif input_option == "Paste URL":
    url = st.sidebar.text_input("Paste URL to a .txt file (e.g., Gutenberg)")
    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            uploaded_text = response.text
            st.sidebar.success("✅ Text fetched successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to fetch from URL: {e}")

# Process Button
process_btn = st.sidebar.button("🔄 Process & Store")

# Handle text processing
if process_btn and uploaded_text:
    with st.spinner("🔄 Processing text and creating ontology vectorstore..."):
        try:
            ontology_data = extract_ontology(uploaded_text)
            docs = ontology_to_documents(ontology_data)
            create_vector_db(docs)
            st.success("✅ Ontology extracted and stored locally!")
        except Exception as e:
            st.error(f"❌ Error processing text: {e}")


# Load DB
try:
    db = load_vector_db()
    qa_chain = create_qa_chain(db)
except:
    st.error("❌ No database found. Upload and process a book first.")

# QA Section
if qa_chain:
    st.markdown("## 💬 Ask a Question")
    query = st.text_input("Enter your question here:")
    if query:
        with st.spinner("💡 Thinking..."):
            result = qa_chain(query)
            st.write("### 📖 Answer")
            st.success(result['result'])

