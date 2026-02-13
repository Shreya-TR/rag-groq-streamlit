import json
from datetime import datetime

import streamlit as st

from config import APP_TITLE, CHUNK_OVERLAP, CHUNK_SIZE, TOP_K_RETRIEVE
from main import build_index, generate_eval_set, run_query, summarize
from rag.eval import pass_fail, retrieval_metrics
from rag.ingest import ingest_files

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .hero {padding: 1rem 1.2rem; border-radius: 14px; background: linear-gradient(120deg, #0b132b, #1c2541, #3a506b); color: white;}
    .kpi {padding: 0.8rem 1rem; border: 1px solid #d9e2f3; border-radius: 12px; background: #f9fbff;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h2 style="margin:0;">Enterprise Copilot + Analyst Multimodal RAG</h2>
      <p style="margin:0.3rem 0 0 0;">Dense Embeddings + Retrieval + Reranking + Grounded Generation + Evidence Trace</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Runtime Config")
    top_k = st.slider("Top-K retrieval", 3, 12, TOP_K_RETRIEVE)
    answer_mode = st.radio("Answer mode", options=["detailed", "concise", "executive"], horizontal=True)
    modality_filter = st.selectbox("Modality filter", options=["all", "text", "image", "audio"])
    st.caption(f"Chunk defaults: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    st.caption("Required secrets: GROQ_API_KEY, JINA_API_KEY")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = ""
if "eval_questions" not in st.session_state:
    st.session_state.eval_questions = []
if "ingest_latency" not in st.session_state:
    st.session_state.ingest_latency = {}
if "ingest_stats" not in st.session_state:
    st.session_state.ingest_stats = {}

uploaded_files = st.file_uploader(
    "Upload source files",
    type=["pdf", "png", "jpg", "jpeg", "wav", "mp3", "m4a"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Process & Build Index", type="primary"):
    with st.spinner("Ingesting files and building enterprise index..."):
        chunks, latency, stats = ingest_files(uploaded_files)
        st.session_state.ingest_latency = latency
        st.session_state.ingest_stats = stats
        if not chunks:
            st.session_state.chunks = []
            st.session_state.index_ready = False
            st.error("No extractable content found in uploaded files.")
        else:
            build_index(chunks)
            st.session_state.chunks = chunks
            st.session_state.index_ready = True
            st.session_state.chat_history = []
            st.session_state.doc_summary = summarize(chunks)
            st.session_state.eval_questions = generate_eval_set(chunks, n=5)
            st.success(f"Indexed {len(chunks)} chunks successfully.")

if st.session_state.index_ready:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi'><b>Total chunks</b><br>{len(st.session_state.chunks)}</div>", unsafe_allow_html=True)
    c2.markdown(
        f"<div class='kpi'><b>Text/Image/Audio</b><br>{st.session_state.ingest_stats.get('text',0)}/"
        f"{st.session_state.ingest_stats.get('image',0)}/{st.session_state.ingest_stats.get('audio',0)}</div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div class='kpi'><b>Ingest latency</b><br>{st.session_state.ingest_latency.get('ingest_ms', 0)} ms</div>",
        unsafe_allow_html=True,
    )
    c4.markdown(
        f"<div class='kpi'><b>Indexed at</b><br>{datetime.now().strftime('%H:%M:%S')}</div>",
        unsafe_allow_html=True,
    )

tab_copilot, tab_analyst = st.tabs(["Copilot Studio", "Analyst Lab"])

with tab_copilot:
    if st.session_state.doc_summary:
        with st.expander("Document Brief", expanded=False):
            st.write(st.session_state.doc_summary)

    q = st.text_input("Ask a grounded question")
    if q:
        if not st.session_state.index_ready:
            st.info("Upload files and build index first.")
        else:
            with st.spinner("Running retrieve -> rerank -> answer..."):
                answer_result, evidence, latency = run_query(
                    q,
                    filters=modality_filter,
                    top_k=top_k,
                    mode=answer_mode,
                    history=st.session_state.chat_history,
                )
            metrics = retrieval_metrics(evidence)
            st.subheader("Answer")
            st.write(answer_result.answer)
            st.caption(f"Confidence: {answer_result.confidence:.3f}")
            st.caption(f"Citations: {' '.join([f'[{e.chunk_id}]' for e in evidence])}")

            with st.expander("Evidence Trace"):
                for i, e in enumerate(evidence, start=1):
                    st.markdown(
                        f"**#{i} {e.chunk_id}** | modality={e.modality} | source={e.source_name} | "
                        f"retrieve={e.score:.3f} | rerank={e.rerank_score:.3f}"
                    )
                    st.write(e.text[:1400] + ("..." if len(e.text) > 1400 else ""))

            with st.expander("Latency + Quality Diagnostics"):
                st.write(latency)
                st.write(metrics)

            st.session_state.chat_history.append(
                {
                    "question": q,
                    "answer": answer_result.answer,
                    "confidence": f"{answer_result.confidence:.3f}",
                    "citations": answer_result.citations,
                }
            )

            report = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "query": q,
                "answer": answer_result.answer,
                "confidence": answer_result.confidence,
                "model_id": answer_result.model_id,
                "latency_ms": latency,
                "retrieval_metrics": metrics,
                "citations": answer_result.citations,
                "evidence": [
                    {
                        "chunk_id": e.chunk_id,
                        "modality": e.modality,
                        "source_name": e.source_name,
                        "retrieval_score": e.score,
                        "rerank_score": e.rerank_score,
                    }
                    for e in evidence
                ],
            }
            st.download_button(
                "Download JSON Report",
                data=json.dumps(report, indent=2),
                file_name="enterprise_rag_report.json",
                mime="application/json",
            )
            st.download_button(
                "Download TXT Report",
                data=f"Query: {q}\n\nAnswer:\n{answer_result.answer}\n\nConfidence: {answer_result.confidence:.3f}",
                file_name="enterprise_rag_report.txt",
                mime="text/plain",
            )

    if st.session_state.chat_history:
        st.subheader("Session Memory")
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()
        for i, turn in enumerate(st.session_state.chat_history[-6:], start=1):
            st.markdown(f"**Q{i}:** {turn['question']}")
            st.markdown(f"**A{i}:** {turn['answer']}")
            st.caption(f"confidence={turn['confidence']} | citations={turn['citations']}")

with tab_analyst:
    st.markdown("### Benchmark Pack")
    if not st.session_state.index_ready:
        st.info("Build index first to run analyst workflows.")
    else:
        col_l, col_r = st.columns([1, 1])
        with col_l:
            if st.button("Refresh Benchmark Questions"):
                st.session_state.eval_questions = generate_eval_set(st.session_state.chunks, n=6)
                st.rerun()
        with col_r:
            run_benchmark = st.button("Run Benchmark Pack", type="primary")

        if run_benchmark and st.session_state.eval_questions:
            rows = []
            for eq in st.session_state.eval_questions:
                ans, ev, lat = run_query(eq, filters=modality_filter, top_k=top_k, mode="concise", history=[])
                verdict = pass_fail(ans.confidence, threshold=0.30)
                rows.append(
                    {
                        "question": eq,
                        "confidence": round(ans.confidence, 3),
                        "avg_rerank": retrieval_metrics(ev)["avg_score"],
                        "latency_ms": lat.get("total_query_ms", 0),
                        "verdict": verdict,
                    }
                )
            st.dataframe(rows, use_container_width=True)

        st.markdown("### Side-by-Side Modality Comparison")
        cmp_query = st.text_input("Comparison query", key="cmp_query")
        left_filter = st.selectbox("Left filter", ["all", "text", "image", "audio"], key="left_filter")
        right_filter = st.selectbox("Right filter", ["all", "text", "image", "audio"], index=1, key="right_filter")
        if cmp_query:
            c1, c2 = st.columns(2)
            with c1:
                ans_l, ev_l, _ = run_query(cmp_query, filters=left_filter, top_k=top_k, mode="concise", history=[])
                st.markdown(f"**{left_filter.upper()}** | confidence={ans_l.confidence:.3f}")
                st.write(ans_l.answer)
                st.caption(f"Top citation: {ev_l[0].chunk_id if ev_l else 'none'}")
            with c2:
                ans_r, ev_r, _ = run_query(cmp_query, filters=right_filter, top_k=top_k, mode="concise", history=[])
                st.markdown(f"**{right_filter.upper()}** | confidence={ans_r.confidence:.3f}")
                st.write(ans_r.answer)
                st.caption(f"Top citation: {ev_r[0].chunk_id if ev_r else 'none'}")
