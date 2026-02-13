import os
from datetime import datetime

import streamlit as st

from main import (
    build_faiss_index,
    chunk_text,
    extract_audio_text,
    extract_image_text,
    extract_pdf_text,
    generate_answer,
    retrieve_with_scores,
    save_uploaded_to_temp,
    summarize_document,
)

st.set_page_config(page_title="Multimodal RAG with Groq", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {padding: 1rem 1.2rem; border-radius: 14px; background: linear-gradient(135deg, #0f172a, #1e3a8a); color: white;}
    .metric-card {padding: 0.8rem 1rem; border-radius: 10px; border: 1px solid #dbeafe; background: #f8fbff;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h2 style="margin:0;">Advanced Multimodal RAG</h2>
      <p style="margin:0.3rem 0 0 0;">PDF + Image OCR + Audio Transcription -> Overlap Chunking -> Retrieval -> Groq Answering</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuration")
    chunk_size = st.slider("Chunk size (words)", min_value=150, max_value=700, value=350, step=25)
    overlap = st.slider("Chunk overlap (words)", min_value=0, max_value=200, value=70, step=10)
    top_k = st.slider("Top chunks for retrieval", min_value=2, max_value=8, value=4)
    st.markdown("---")
    st.markdown("Supported: `pdf`, `png`, `jpg`, `jpeg`, `wav`, `mp3`, `m4a`")
    if not os.getenv("GROQ_API_KEY"):
        st.warning("`GROQ_API_KEY` missing: app will run in retrieval-only fallback mode.")

uploaded_files = st.file_uploader(
    "Upload one or more files",
    type=["pdf", "png", "jpg", "jpeg", "wav", "mp3", "m4a"],
    accept_multiple_files=True,
)

if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "corpus_text" not in st.session_state:
    st.session_state.corpus_text = ""
if "ingest_time" not in st.session_state:
    st.session_state.ingest_time = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = ""

if uploaded_files and st.button("Process Files", type="primary"):
    with st.spinner("Extracting text and building retrieval index..."):
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
                    extracted = extract_audio_text(temp_path)
            except Exception as exc:
                st.error(f"Failed processing `{file.name}`: {exc}")

            if extracted.strip():
                combined_text.append(extracted)
                processed += 1

        full_text = "\n\n".join(combined_text).strip()
        if not full_text:
            st.session_state.index = None
            st.session_state.chunks = []
            st.session_state.corpus_text = ""
            st.session_state.doc_summary = ""
            st.error("No text could be extracted from uploaded files.")
        else:
            chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                st.session_state.index = None
                st.session_state.chunks = []
                st.session_state.corpus_text = ""
                st.session_state.doc_summary = ""
                st.error("Chunking produced no chunks. Try reducing overlap or uploading more text.")
            else:
                index, chunks = build_faiss_index(chunks)
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.corpus_text = full_text
                st.session_state.ingest_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.chat_history = []
                st.session_state.doc_summary = summarize_document(chunks)
                st.success(f"Processed {processed} file(s) and built {len(chunks)} chunks.")

if st.session_state.chunks:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='metric-card'><b>Total chunks</b><br>{len(st.session_state.chunks)}</div>", unsafe_allow_html=True)
    with c2:
        total_words = len(st.session_state.corpus_text.split())
        st.markdown(f"<div class='metric-card'><b>Total words</b><br>{total_words}</div>", unsafe_allow_html=True)
    with c3:
        ingest_time = st.session_state.ingest_time or "-"
        st.markdown(f"<div class='metric-card'><b>Indexed at</b><br>{ingest_time}</div>", unsafe_allow_html=True)

if st.session_state.doc_summary:
    with st.expander("Document Summary", expanded=False):
        st.write(st.session_state.doc_summary)

query = st.text_input("Ask a question from your uploaded content")
if query:
    if st.session_state.index is None:
        st.info("Upload and process files first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            sources = retrieve_with_scores(query, st.session_state.index, st.session_state.chunks, top_k=top_k)
            answer = generate_answer(
                query,
                st.session_state.index,
                st.session_state.chunks,
                history=st.session_state.chat_history,
            )

        citations = [f"[Chunk {chunk_idx + 1} | score={score:.3f}]" for _, score, chunk_idx in sources]
        st.subheader("Answer")
        st.write(answer)
        st.caption("Citations: " + " ".join(citations))

        with st.expander("Retrieved Sources"):
            for rank, (src, score, chunk_idx) in enumerate(sources, start=1):
                st.markdown(f"**Chunk {chunk_idx + 1} | Rank {rank} | Score {score:.3f}**")
                st.write(src[:1200] + ("..." if len(src) > 1200 else ""))

        st.session_state.chat_history.append(
            {
                "question": query,
                "answer": answer,
                "citations": " ".join(citations),
            }
        )

        report = f"Question: {query}\n\nAnswer:\n{answer}\n\nCitations:\n{' '.join(citations)}\n\n---\nRetrieved Sources:\n\n" + "\n\n".join(
            [f"[Chunk {chunk_idx + 1} | Score {score:.3f}] {text}" for text, score, chunk_idx in sources]
        )
        st.download_button(
            label="Download Q&A Report",
            data=report,
            file_name="rag_answer_report.txt",
            mime="text/plain",
        )

if st.session_state.chat_history:
    st.subheader("Conversation Memory")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    for i, turn in enumerate(st.session_state.chat_history[-8:], start=1):
        st.markdown(f"**Q{i}:** {turn['question']}")
        st.markdown(f"**A{i}:** {turn['answer']}")
        st.caption(turn.get("citations", ""))
