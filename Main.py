import pdfplumber, pytesseract, faiss, whisper
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import streamlit as st

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def extract_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_image_text(path):
    return pytesseract.image_to_string(Image.open(path))

def extract_audio_text(path):
    model = whisper.load_model("base")
    return model.transcribe(path)["text"]

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def retrieve(query, index, chunks, top_k=3):
    q_emb = embedder.encode([query])
    _, idxs = index.search(np.array(q_emb), top_k)
    return [chunks[i] for i in idxs[0]]

def generate_answer(query, index, chunks):
    context = "\n".join(retrieve(query, index, chunks))
    prompt = f"Answer the question based on context:\n{context}\n\nQuestion: {query}"
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content
