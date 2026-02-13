import os
import tempfile
import time
import uuid
from typing import Dict, List, Tuple

import pdfplumber
import pytesseract
from PIL import Image

from config import ASR_MODEL, CHUNK_OVERLAP, CHUNK_SIZE
from rag.chunking import chunk_text
from rag.types import DocumentChunk
from rag.vision import image_to_caption

try:
    from groq import Groq
except ImportError:
    Groq = None


def save_uploaded_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _transcribe_audio(path: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or Groq is None:
        return ""
    client = Groq(api_key=api_key)
    with open(path, "rb") as audio_file:
        tr = client.audio.transcriptions.create(file=audio_file, model=ASR_MODEL, response_format="verbose_json")
    return (getattr(tr, "text", "") or "").strip()


def ingest_files(uploaded_files) -> Tuple[List[DocumentChunk], Dict[str, float], Dict[str, int]]:
    t0 = time.perf_counter()
    chunks: List[DocumentChunk] = []
    stats = {"text": 0, "image": 0, "audio": 0}

    for file in uploaded_files:
        path = save_uploaded_to_temp(file)
        name = file.name
        lname = name.lower()

        if lname.endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                for page_i, page in enumerate(pdf.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if not text:
                        continue
                    for ci, part in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP), start=1):
                        chunks.append(
                            DocumentChunk(
                                id=str(uuid.uuid4()),
                                text=part,
                                modality="text",
                                source_name=name,
                                page_or_ts=f"page-{page_i}",
                                metadata={"chunk_no": ci},
                            )
                        )
                        stats["text"] += 1

        elif lname.endswith((".png", ".jpg", ".jpeg")):
            ocr = pytesseract.image_to_string(Image.open(path)).strip()
            vision = image_to_caption(path)
            merged = "\n".join([s for s in [ocr, vision] if s]).strip()
            if merged:
                for ci, part in enumerate(chunk_text(merged, CHUNK_SIZE, CHUNK_OVERLAP), start=1):
                    chunks.append(
                        DocumentChunk(
                            id=str(uuid.uuid4()),
                            text=part,
                            modality="image",
                            source_name=name,
                            page_or_ts="image-1",
                            metadata={"chunk_no": ci},
                        )
                    )
                    stats["image"] += 1

        elif lname.endswith((".wav", ".mp3", ".m4a")):
            text = _transcribe_audio(path)
            if text:
                for ci, part in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP), start=1):
                    chunks.append(
                        DocumentChunk(
                            id=str(uuid.uuid4()),
                            text=part,
                            modality="audio",
                            source_name=name,
                            page_or_ts="audio-1",
                            metadata={"chunk_no": ci},
                        )
                    )
                    stats["audio"] += 1

    latency = {"ingest_ms": round((time.perf_counter() - t0) * 1000, 2)}
    return chunks, latency, stats
