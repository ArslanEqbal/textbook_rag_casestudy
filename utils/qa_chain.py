# utils/qa_chain.py
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def create_qa_chain(vectorstore):
    llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa
