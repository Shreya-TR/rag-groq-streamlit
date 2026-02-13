import os

import streamlit as st

from main import (
    build_faiss_index,
    chunk_text,
    extract_audio_text,
    extract_image_text,
    extract_pdf_text,
    generate_answer,
    save_uploaded_to_temp,
)

st.set_page_config(page_title="Multimodal RAG with Groq + FAISS", page_icon="📚", layout="wide")

st.title("Advanced Multimodal RAG")
st.caption("PDF + Image OCR + Audio Transcription -> Overlap Chunking -> FAISS Retrieval -> Groq Answering")

with st.sidebar:
    st.header("Configuration")
    chunk_size = st.slider("Chunk size (words)", min_value=150, max_value=700, value=350, step=25)
    overlap = st.slider("Chunk overlap (words)", min_value=0, max_value=200, value=70, step=10)
    st.markdown("---")
    st.markdown("Supported files: `pdf`, `png`, `jpg`, `jpeg`, `wav`, `mp3`, `m4a`")
    if not os.getenv("GROQ_API_KEY"):
        st.warning("`GROQ_API_KEY` not found. Retrieval still works; answer generation falls back.")

uploaded_files = st.file_uploader(
    "Upload one or more files",
    type=["pdf", "png", "jpg", "jpeg", "wav", "mp3", "m4a"],
    accept_multiple_files=True,
)

if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if uploaded_files and st.button("Process Files", type="primary"):
    with st.spinner("Extracting text and building index..."):
        combined_text = []
        processed = 0

        for file in uploaded_files:
            temp_path = save_uploaded_to_temp(file)
            file_name = file.name.lower()
            extracted = ""

            try:
                if file_name.endswith(".pdf"):
                    extracted = extract_pdf_text(temp_path)
                elif file_name.endswith((".png", ".jpg", ".jpeg")):
                    extracted = extract_image_text(temp_path)
                elif file_name.endswith((".mp3", ".wav", ".m4a")):
                    extracted = extract_audio_text(temp_path, model_size="base")
            except Exception as exc:
                st.error(f"Failed processing `{file.name}`: {exc}")

            if extracted.strip():
                combined_text.append(extracted)
                processed += 1

        full_text = "\n\n".join(combined_text).strip()
        if not full_text:
            st.session_state.index = None
            st.session_state.chunks = []
            st.error("No text could be extracted from uploaded files.")
        else:
            chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                st.session_state.index = None
                st.session_state.chunks = []
                st.error("Chunking produced no chunks. Try reducing overlap or uploading more text.")
            else:
                index, chunks = build_faiss_index(chunks)
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.success(f"Processed {processed} file(s). Built {len(chunks)} chunks.")

query = st.text_input("Ask a question from your uploaded content")
if query:
    if st.session_state.index is None:
        st.info("Upload and process files first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            answer = generate_answer(query, st.session_state.index, st.session_state.chunks)
        st.subheader("Answer")
        st.write(answer)
