import os


APP_TITLE = "Enterprise Multimodal RAG"

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v4")
VISION_MODEL = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
ASR_MODEL = os.getenv("ASR_MODEL", "whisper-large-v3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "jina-reranker-v2-base-multilingual")

# Retrieval defaults
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "8"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "90"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "10000"))

# Confidence thresholds
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.25"))

# Persistence
INDEX_STORE_DIR = os.getenv("INDEX_STORE_DIR", ".rag_store")
