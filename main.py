# main.py
import os, io, time, json, re
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import trafilatura
from pypdf import PdfReader
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
import openai

# ========== Configuration (use env vars in production) ==========
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")       # optional if you use NewsAPI ingestion
OPENAI_KEY = os.getenv("OPENAI_API_KEY")     # optional for RAG answers
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss.index")
FAISS_META_PATH = os.getenv("FAISS_META_PATH", "faiss_meta.json")

if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# ========== Utilities ==========
def simple_split(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) <= chunk_size:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(s) > chunk_size:
                for i in range(0, len(s), chunk_size - overlap):
                    chunks.append(s[i:i+chunk_size])
                cur = ""
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return chunks

# ========== Embedding model & FAISS wrapper ==========
class FaissStore:
    def __init__(self):
        self.index = None
        self.meta = []  # list of dicts
        self.dim = None

    def init_if_needed(self, dim):
        if self.index is None:
            self.dim = dim
            self.index = faiss.IndexFlatIP(dim)

    def add(self, embs: np.ndarray, metadatas: List[dict]):
        if embs.shape[0] == 0:
            return
        self.init_if_needed(embs.shape[1])
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        self.index.add(embs_norm)
        self.meta.extend(metadatas)

    def search(self, q_emb: np.ndarray, k=5):
        if self.index is None or self.index.ntotal == 0:
            return []
        qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
        D, I = self.index.search(qn, k)
        res = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0: continue
            m = self.meta[int(idx)].copy()
            m["score"] = float(score)
            res.append(m)
        return res

    def save(self, index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH):
        if self.index is not None:
            faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def load(self, index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH):
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.dim = self.index.d
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

# initialize model & store
embed_model = SentenceTransformer(EMBED_MODEL)
store = FaissStore()
store.load()

# ========== FastAPI app ==========
app = FastAPI(title="Equity News Research API")

# Allow calls from your React UI (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Pydantic schemas ==========
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    rag: Optional[bool] = False

class IngestUrlsRequest(BaseModel):
    urls: List[str]

# ========== Endpoints ==========
@app.get("/status")
def status():
    return {"ready": True, "index_size": len(store.meta), "embed_model": EMBED_MODEL}

@app.post("/ingest/files")
async def ingest_files(files: List[UploadFile] = File(...)):
    """
    Accept multiple files (.pdf, .txt, .csv). Returns ingested count & metadata.
    """
    ingested = []
    for f in files:
        filename = f.filename.lower()
        content = await f.read()
        try:
            if filename.endswith(".pdf"):
                reader = PdfReader(io.BytesIO(content))
                pages = []
                for p in reader.pages:
                    pages.append(p.extract_text() or "")
                text = "\n".join(pages).strip()
                title = filename
            elif filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
                # If CSV has a text-like column 'text' or 'content' use it; otherwise convert rows to text
                if "text" in df.columns:
                    texts = df["text"].astype(str).tolist()
                    text = "\n\n".join(texts)
                else:
                    text = df.astype(str).to_csv(index=False)
                title = filename
            else:
                # treat as plain text
                text = content.decode("utf-8", errors="ignore")
                title = filename
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read {filename}: {str(e)}")

        if not text or len(text.strip()) == 0:
            continue
        # chunk
        parts = simple_split(text)
        texts = parts
        # embed in batches
        embs = embed_model.encode(texts, convert_to_numpy=True)
        metas = []
        for i, p in enumerate(parts):
            metas.append({"title": title, "source": "upload", "chunk_id": f"{filename}_{i}", "text": p})
        store.add(embs, metas)
        ingested.append({"title": title, "chunks": len(parts)})

    store.save()
    return {"ingested": ingested, "total_indexed": len(store.meta)}

@app.post("/ingest/urls")
def ingest_urls(req: IngestUrlsRequest):
    """
    Accepts JSON with `urls: [...]`. Extracts with trafilatura, indexes.
    """
    ingested = []
    for url in req.urls:
        try:
            html = trafilatura.fetch_url(url)
            if not html:
                continue
            text = trafilatura.extract(html)
            if not text:
                continue
            parts = simple_split(text)
            embs = embed_model.encode(parts, convert_to_numpy=True)
            metas = []
            for i, p in enumerate(parts):
                metas.append({"title": None, "source": url, "chunk_id": f"{int(time.time())}_{i}", "text": p})
            store.add(embs, metas)
            ingested.append({"url": url, "chunks": len(parts)})
        except Exception as e:
            # skip and continue
            continue
    store.save()
    return {"ingested": ingested, "total_indexed": len(store.meta)}

@app.post("/ingest/newsapi")
def ingest_newsapi(query: str = Form(...), max_articles: int = Form(10)):
    """
    Use NewsAPI to fetch recent articles (requires NEWSAPI_KEY env var).
    """
    if not NEWSAPI_KEY:
        raise HTTPException(status_code=400, detail="NEWSAPI_KEY not configured")
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": max_articles}
    headers = {"Authorization": NEWSAPI_KEY}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    ingested = []
    for a in articles:
        u = a.get("url")
        try:
            html = trafilatura.fetch_url(u)
            txt = trafilatura.extract(html) if html else ""
            if not txt: continue
            parts = simple_split(txt)
            embs = embed_model.encode(parts, convert_to_numpy=True)
            metas = []
            for i, p in enumerate(parts):
                metas.append({"title": a.get("title"), "source": a.get("source", {}).get("name"), "url": u, "chunk_id": f"news_{int(time.time())}_{i}", "text": p})
            store.add(embs, metas)
            ingested.append({"url": u, "title": a.get("title"), "chunks": len(parts)})
        except Exception:
            continue
    store.save()
    return {"ingested": ingested, "total_indexed": len(store.meta)}

@app.post("/query")
def query_endpoint(req: QueryRequest):
    q = req.query
    k = req.top_k or 5
    if len(store.meta) == 0:
        raise HTTPException(status_code=400, detail="Index empty. Ingest some data first.")
    q_emb = embed_model.encode([q], convert_to_numpy=True)
    results = store.search(q_emb, k=k)
    output = {"answers": results}
    # optional RAG via OpenAI
    if req.rag and OPENAI_KEY:
        # craft context using top results
        context = "\n\n---\n\n".join([f"CHUNK_ID: {r.get('chunk_id','n/a')} | SOURCE: {r.get('source')}\n{r.get('text')}" for r in results])
        prompt = f"""You are an equity research assistant. Use only the facts in context. Context:\n{context}\n\nQuestion: {q}\nProvide: short summary, 3 supporting bullets with source chunk IDs, confidence."""
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a factual financial research assistant."},{"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0.0
        )
        output["rag_answer"] = resp["choices"][0]["message"]["content"]
    return output
