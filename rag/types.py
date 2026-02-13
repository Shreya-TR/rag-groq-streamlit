from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    id: str
    text: str
    modality: str
    source_name: str
    page_or_ts: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    rerank_score: float
    modality: str
    source_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalIndex:
    chunks: List[DocumentChunk]
    embeddings: Any
    vector_index: Any
    embedding_model: str
    index_kind: str


@dataclass
class AnswerResult:
    answer: str
    citations: List[str]
    latency_ms: Dict[str, float]
    confidence: float
    model_id: str
