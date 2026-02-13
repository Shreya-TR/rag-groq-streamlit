import streamlit as st
from main import extract_pdf_text, extract_image_text, extract_audio_text, chunk_text, build_faiss_index, generate_answer

st.title("📚 Multimodal RAG Demo")
st.write("Upload PDFs, images, or audio files, then ask questions!")

uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
index, chunks = None, []

if uploaded_files:
    combined_text = ""
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.read())
        if file.name.endswith(".pdf"):
            combined_text += extract_pdf_text(file.name)
        elif file.name.endswith((".png", ".jpg", ".jpeg")):
            combined_text += extract_image_text(file.name)
        elif file.name.endswith((".mp3", ".wav", ".m4a")):
            combined_text += extract_audio_text(file.name)

    chunks = chunk_text(combined_text)
    index, chunks = build_faiss_index(chunks)

query = st.text_input("Enter your question:")
if query and index:
    st.markdown("### Answer")
    st.write(generate_answer(query, index, chunks))
