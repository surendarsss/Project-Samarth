# streamlit_rag_openrouter.py
import os
import json
import time
from pathlib import Path

import streamlit as st

# optional heavy libs (we'll import after download check)
try:
    import faiss
except Exception:
    faiss = None
import numpy as np
import sqlite3

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Optional OpenRouter client via openai package
try:
    from openai import OpenAI
    OPENAI_PKG_AVAILABLE = True
except Exception:
    OPENAI_PKG_AVAILABLE = False

load_dotenv()

# ----------------- Config -----------------
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")

# local paths (after downloading we store in data/)
LOCAL_DIR = Path("data")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(LOCAL_DIR / "faiss_index.bin")))
META_DB_PATH = Path(os.getenv("META_DB_PATH", str(LOCAL_DIR / "meta.sqlite")))

# HF dataset settings (set HF_DATASET_REPO in env or secrets)
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "surendarssss/project_smarth")
HF_INDEX_FILENAME = os.getenv("HF_INDEX_FILENAME", "faiss_index.bin")
HF_META_FILENAME = os.getenv("HF_META_FILENAME", "meta.sqlite")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
TOP_K = int(os.getenv("TOP_K", 20))

# ----------------- Helper: download from HF dataset -----------------
def download_from_hf_dataset():
    """
    Download faiss_index.bin and meta.sqlite from HF dataset repo to local data/ directory.
    Uses HF_TOKEN from environment for private datasets. Returns (success: bool, msg: str).
    """
    token = os.getenv("streamlit")  # set this in Streamlit Secrets or HF Space secrets
    if not HF_DATASET_REPO:
        return False, "HF_DATASET_REPO not set."

    try:
        st.info(f"Downloading {HF_INDEX_FILENAME} and {HF_META_FILENAME} from {HF_DATASET_REPO} ...")
        idx_cache = hf_hub_download(repo_id=HF_DATASET_REPO, filename=HF_INDEX_FILENAME, repo_type="dataset", token=token)
        meta_cache = hf_hub_download(repo_id=HF_DATASET_REPO, filename=HF_META_FILENAME, repo_type="dataset", token=token)
    except Exception as e:
        return False, f"hf_hub_download failed: {e}"

    # copy to local paths atomically
    try:
        tmp_idx = FAISS_INDEX_PATH.with_suffix(".tmp")
        tmp_meta = META_DB_PATH.with_suffix(".tmp")
        tmp_idx.parent.mkdir(parents=True, exist_ok=True)
        # copy (hf_hub_download gives local cached paths; use shutil.copy2 if you want)
        tmp_idx.write_bytes(Path(idx_cache).read_bytes())
        tmp_meta.write_bytes(Path(meta_cache).read_bytes())
        tmp_idx.replace(FAISS_INDEX_PATH)
        tmp_meta.replace(META_DB_PATH)
        return True, "Downloaded and stored locally."
    except Exception as e:
        return False, f"Failed to copy downloaded files: {e}"

# ----------------- Resource loader (with HF download) -----------------
@st.cache_resource(show_spinner=False)
def load_index_and_meta_with_download():
    # attempt download from HF dataset if files not present
    if not FAISS_INDEX_PATH.exists() or not META_DB_PATH.exists():
        ok, msg = download_from_hf_dataset()
        if not ok:
            raise FileNotFoundError(f"Could not download artifacts from HF dataset: {msg}")

    # now load
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
    if not META_DB_PATH.exists():
        raise FileNotFoundError(f"Meta DB not found at {META_DB_PATH}")

    # load faiss
    try:
        index = faiss.read_index(str(FAISS_INDEX_PATH))
    except Exception as e:
        raise RuntimeError(f"Failed to read FAISS index: {e}")

    # load metadata
    conn = sqlite3.connect(str(META_DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    if "documents" not in tables:
        raise RuntimeError(f"'documents' table not found in meta DB. Available: {tables}")

    cur.execute("SELECT vector_id, doc_id, text, source_file, created_at FROM documents WHERE active=1")
    rows = cur.fetchall()
    meta_map = {}
    for row in rows:
        vid = int(row[0])
        meta_map[vid] = {"doc_id": row[1], "text": row[2], "source_file": row[3], "created_at": row[4]}

    # load embedder
    embed_model = SentenceTransformer(EMB_MODEL)
    return index, meta_map, conn, embed_model

# ----------------- Load resources -----------------
try:
    index, meta_map, meta_conn, embed_model = load_index_and_meta_with_download()
except Exception as e:
    st.error(f"Failed to load index/meta: {e}")
    st.stop()

# ----------------- rest of your helpers (unchanged) -----------------
def embed_query(q: str):
    v = embed_model.encode([q], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(v)
    return v

def retrieve_topk(index, meta_map, qvec, k=TOP_K):
    D, I = index.search(qvec, k)
    results = []
    for score, vid in zip(D[0], I[0]):
        if vid == -1:
            continue
        m = meta_map.get(int(vid))
        if not m:
            continue
        results.append({"vector_id": int(vid), "score": float(score), **m})
    return results

def build_prompt(question: str, docs: list):
    """
    Build a cleaner prompt for LLM — context included, but doc IDs hidden in final display.
    """
    ctx_parts = []
    for d in docs:
        text = d["text"]
        # truncate overly long context
        if len(text) > 2000:
            text = text[:2000] + " ...[truncated]"
        # keep only text, hide doc ids
        ctx_parts.append(text)

    ctx = "\n\n".join(ctx_parts) if ctx_parts else "No context available."

    system_msg = (
        "You are an expert analyst. Use ONLY the CONTEXT below to answer the question. "
        "Do NOT include any document IDs, filenames, or citations in your final answer. "
        "If the answer cannot be found, say 'Insufficient data in provided documents.'"
    )

    prompt = f"""{system_msg}

CONTEXT:
{ctx}

QUESTION:
{question}

Answer concisely and naturally. Do not include any [DOC id] references or metadata in the answer.
"""
    return prompt


# OpenRouter call helper (using openai.OpenAI client)
def call_openrouter(prompt: str):
    if not OPENAI_PKG_AVAILABLE:
        raise RuntimeError("openai package not installed in this environment.")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not configured in environment (.env).")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    completion = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role":"system","content":"You are a helpful RAG-based assistant."},
                  {"role":"user","content":prompt}],
        extra_headers={"HTTP-Referer":"http://localhost:8501", "X-Title":"RAG Streamlit App"},
        temperature=0.0,
        max_tokens=800,
    )
    try:
        return completion.choices[0].message.content
    except Exception:
        return str(completion)

# Optional logo (place your logo file in same directory or give URL)
logo_path = "logo.png"  # or use a link: "https://huggingface.co/front/assets/huggingface_logo.svg"

st.set_page_config(page_title="Project Smarth - RAG Q&A", layout="centered")

# Top header
col1, col2 = st.columns([1, 4])
with col1:
    if Path(logo_path).exists():
        st.image(logo_path, width=80)
with col2:
    st.markdown(
        """
        <h2 style="margin-bottom: 0;">🚦 Project Smarth</h2>
        <p style="color:gray; margin-top:0;">AI-powered Intelligent Q&A System</p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Input field
st.markdown("### Ask a Question")
question = st.text_input(
    "Type your question below 👇 (Example: 'What is the average rainfall in Kerala 2015-2020?')",
    placeholder="Enter your question...",
)

# Ask button
if st.button("Ask"):
    if not question.strip():
        st.warning("⚠️ Please enter a question first.")
    else:
        with st.spinner("Processing your question..."):
            qvec = embed_query(question)
            results = retrieve_topk(index, meta_map, qvec, k=TOP_K)

        if not results:
            st.info("No relevant documents found. Try a different question.")
        else:
            prompt = build_prompt(question, results)
            st.markdown("---")
            st.subheader("💡 AI Answer")
            try:
                if OPENROUTER_API_KEY and OPENAI_PKG_AVAILABLE:
                    with st.spinner("Getting answer from OpenRouter..."):
                        answer = call_openrouter(prompt)
                    st.success("Answer generated successfully ✅")
                    st.markdown(
    f"<div style='background-color:#f8f9fa; padding:15px; border-radius:10px; font-size:16px; color:#222;'>"
    f"{answer}"
    f"</div>",
    unsafe_allow_html=True,
)
                else:
                    st.warning("OpenRouter not configured — showing prompt only.")
                    st.code(prompt[:4000])
            except Exception as e:
                st.error("❌ LLM call failed: " + str(e))
                st.code(prompt[:2000])

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:gray;">
        <p>Built with ❤️ by <b>Surendar</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)