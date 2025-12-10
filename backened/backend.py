# backend/backend.py
import os
import io
import json
import time
import re
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
import trafilatura
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import uvicorn

# ---------- CONFIG ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss.index")
FAISS_META_PATH = os.getenv("FAISS_META_PATH", "faiss_meta.json")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
# ----------------------------

# ---------- Helpers ----------
def split_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) <= size:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(s) > size:
                for i in range(0, len(s), size - overlap):
                    chunks.append(s[i:i+size])
                cur = ""
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return chunks

# ---------- FAISS store ----------
class FaissStore:
    def __init__(self):
        self.index = None
        self.metadata: List[dict] = []
        self.dim = None

    def init_if_needed(self, dim):
        if self.index is None:
            self.dim = dim
            self.index = faiss.IndexFlatIP(dim)

    def add(self, embs: np.ndarray, metadatas: List[dict]):
        if embs is None or embs.shape[0] == 0:
            return
        self.init_if_needed(embs.shape[1])
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        self.index.add(embs_norm.astype("float32"))
        self.metadata.extend(metadatas)

    def search(self, q_emb: np.ndarray, k=5):
        if self.index is None or self.index.ntotal == 0:
            return []
        qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
        D, I = self.index.search(qn.astype("float32"), k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            md = self.metadata[int(idx)].copy()
            md["score"] = float(score)
            out.append(md)
        return out

    def save(self, index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH):
        if self.index is not None:
            faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH):
        # load metadata first
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print(f"[store] Loaded metadata ({len(self.metadata)}) from {meta_path}")
        else:
            print(f"[store] No metadata file at {meta_path}")

        # load faiss index if available
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
                self.dim = self.index.d
                print(f"[store] Loaded Faiss index from {index_path} (dim={self.dim})")
            except Exception as e:
                print("[store] Failed to load faiss index:", e)
                self.index = None
        else:
            print(f"[store] No faiss index file at {index_path}")

# ---------- Init model & store ----------
print("[startup] Loading embedding model:", EMBED_MODEL)
embed_model = SentenceTransformer(EMBED_MODEL)
store = FaissStore()
store.load()

# ---------- FastAPI ----------
app = FastAPI(title="Equity Research Backend (Simple)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- Schemas ----------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class UrlsReq(BaseModel):
    urls: List[str]

# ---------- Health ----------
@app.get("/status")
def status():
    return {"ready": True, "indexed_chunks": len(store.metadata), "embed_model": EMBED_MODEL}

# ---------- Ingest files ----------
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    """
    POST multipart/form-data with field 'files' (multiple).
    Supports: .pdf, .csv, .txt
    CSV is ingested row-wise for better retrieval.
    """
    added = 0
    for f in files:
        name = f.filename
        b = await f.read()
        text = ""
        try:
            if name.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(b))
                pages = []
                for p in reader.pages:
                    pages.append(p.extract_text() or "")
                text = "\n".join(pages)
            elif name.lower().endswith(".csv"):
                # Improved CSV ingestion: each row becomes one chunk (preferred columns combined)
                try:
                    df = pd.read_csv(io.BytesIO(b))
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

                preferred_cols = ["title","name","movie","overview","description","plot","summary","text","content"]
                found = [c for c in df.columns if c.lower() in preferred_cols]

                parts = []
                if len(found) > 0:
                    for _, row in df.iterrows():
                        pieces = []
                        for c in found:
                            val = row.get(c, "")
                            if pd.notna(val) and str(val).strip():
                                pieces.append(f"{c}: {val}")
                        if not pieces:
                            for c in df.columns:
                                v = row.get(c, "")
                                if pd.notna(v) and str(v).strip():
                                    pieces.append(f"{c}: {v}")
                        parts.append(" | ".join(pieces))
                else:
                    for _, row in df.iterrows():
                        pieces = []
                        for c in df.columns:
                            v = row.get(c, "")
                            if pd.notna(v) and str(v).strip():
                                pieces.append(f"{c}: {v}")
                        parts.append(" | ".join(pieces))

                if len(parts) > 0:
                    embs = embed_model.encode(parts, convert_to_numpy=True)
                    metas = []
                    ts = int(time.time())
                    for i, chunk_text in enumerate(parts):
                        metas.append({
                            "title": name,
                            "url": None,
                            "chunk_id": f"{name}_{ts}_{i}",
                            "text": chunk_text
                        })
                    store.add(embs, metas)
                    added += len(parts)
                # done with this file
                continue

            else:
                text = b.decode("utf-8", errors="ignore")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read {name}: {e}")

        if not text or len(text.strip()) == 0:
            continue

        parts = split_text(text)
        if len(parts) == 0:
            continue

        embs = embed_model.encode(parts, convert_to_numpy=True)
        metas = []
        ts = int(time.time())
        for i, chunk in enumerate(parts):
            metas.append({
                "title": name,
                "url": None,
                "chunk_id": f"{name}_{ts}_{i}",
                "text": chunk
            })
        store.add(embs, metas)
        added += len(parts)

    store.save()
    return {"status": "ok", "added_chunks": added, "total_indexed": len(store.metadata)}

# ---------- Ingest URLs ----------
@app.post("/ingest/urls")
def ingest_urls(req: UrlsReq):
    added = 0
    for url in req.urls:
        html = trafilatura.fetch_url(url)
        if not html:
            continue
        text = trafilatura.extract(html)
        if not text:
            continue
        parts = split_text(text)
        if not parts:
            continue
        embs = embed_model.encode(parts, convert_to_numpy=True)
        metas = []
        ts = int(time.time())
        for i, chunk in enumerate(parts):
            metas.append({
                "title": None,
                "url": url,
                "chunk_id": f"url_{ts}_{i}",
                "text": chunk
            })
        store.add(embs, metas)
        added += len(parts)
    store.save()
    return {"status": "ok", "added_chunks": added, "total_indexed": len(store.metadata)}

# ---------- Query ----------
@app.post("/query")
def query_endpoint(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="query missing")
    if len(store.metadata) == 0:
        raise HTTPException(status_code=400, detail="index empty; ingest first")
    q_emb = embed_model.encode([req.query], convert_to_numpy=True)
    results = store.search(q_emb, k=req.top_k)
    return {"answers": results}

# ---------- Meta endpoint ----------
@app.get("/meta")
def get_meta(n: int = Query(20, ge=1, le=500)):
    total = len(store.metadata)
    if total == 0:
        return {"total": 0, "items": []}
    items = store.metadata[-n:][::-1]  # newest first
    return {"total": total, "items": items}

# ---------- Save / Load endpoints ----------
@app.post("/save")
def save_index():
    store.save()
    return {"status": "saved", "total_indexed": len(store.metadata)}

@app.post("/load")
def load_index():
    store.load()
    return {"status": "loaded", "total_indexed": len(store.metadata)}

# ---------- Run server ----------
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)
