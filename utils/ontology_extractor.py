# File: utils/ontology_extractor.py
import os
import json
import requests
from groq import Client
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

client = Client(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a knowledge extraction assistant. Given a chunk of a textbook, extract structured ontology 
as JSON with the following format:

IMPORTANT POINT: The output must be valid JSON. Do not include any other text or explanations All important information should be covered in the json output.

[
  {
    "concept": "<main concept>",
    "definition": "<medium definition of the concept>",
    "related_terms": ["term1", "term2", "..."]
  }
]

Only respond with valid JSON. Avoid extra explanations."""




def split_text(text, chunk_size=3000, chunk_overlap=250):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def get_ontology_from_chunk(chunk, model="llama3-70b-8192"):
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chunk}
            ],
            model=model,
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("❌ Error in chunk call:", e)
        return None


def extract_ontology(text):
    chunks = split_text(text)
    ontologies = []
    for chunk in chunks:
        result = get_ontology_from_chunk(chunk)
        if result:
            try:
                parsed = json.loads(result)
                ontologies.extend(parsed)
            except json.JSONDecodeError:
                print("⚠️ Skipped chunk due to JSON error")
    return ontologies
