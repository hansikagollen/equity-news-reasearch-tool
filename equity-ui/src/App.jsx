import React, { useState } from "react";

const BACKEND = "http://127.0.0.1:8000"; // adjust if backend runs elsewhere

export default function App() {
  const [uploadStatus, setUploadStatus] = useState("");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [filesPreview, setFilesPreview] = useState([]);
  const [metaList, setMetaList] = useState([]);
  const [urlsText, setUrlsText] = useState("");
  const [csvRowMode, setCsvRowMode] = useState(true);

  const onFilesChange = (e) => {
    setFilesPreview(Array.from(e.target.files).map(f => f.name));
  };

  const uploadFiles = async () => {
    const input = document.getElementById("fileInput");
    const files = input.files;
    if (!files || files.length === 0) {
      alert("Select files to upload.");
      return;
    }
    setUploadStatus("Uploading...");
    const form = new FormData();
    for (let f of files) form.append("files", f);

    try {
      const res = await fetch(`${BACKEND}/ingest`, { method: "POST", body: form });
      const json = await res.json();
      if (json && json.status === "ok") {
        setUploadStatus(`Added ${json.added_chunks} chunks. Total indexed: ${json.total_indexed}`);
      } else {
        setUploadStatus("Upload response: " + JSON.stringify(json));
      }
    } catch (err) {
      setUploadStatus("Upload failed: " + err);
    }
  };

  const ingestUrls = async () => {
    const urls = urlsText.split("\n").map(u => u.trim()).filter(Boolean);
    if (urls.length === 0) { alert("Add at least one URL (one per line)"); return; }
    setUploadStatus("Fetching & ingesting URLs...");
    try {
      const res = await fetch(`${BACKEND}/ingest/urls`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls })
      });
      const j = await res.json();
      setUploadStatus(`Ingested ${j.added_chunks} chunks. Total indexed: ${j.total_indexed}`);
    } catch (e) {
      setUploadStatus("URL ingest failed: " + e);
    }
  };

  const fetchMeta = async () => {
    try {
      const res = await fetch(`${BACKEND}/meta?n=20`);
      const j = await res.json();
      setMetaList(j.items || []);
    } catch (e) {
      alert("Failed to fetch meta: " + e);
    }
  };

  const ask = async () => {
    if (!query.trim()) { alert("Enter a question"); return; }
    setResults([{ loading: true }]);
    try {
      const res = await fetch(`${BACKEND}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, top_k: 6 }),
      });
      const json = await res.json();
      setResults(json.answers || []);
    } catch (err) {
      setResults([{ error: String(err) }]);
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Equity Research Assistant</h1>

      <div style={styles.card}>
        <h3>Upload documents (PDF / CSV / TXT)</h3>
        <div style={{display:"flex", gap:10, alignItems:"center"}}>
          <input id="fileInput" type="file" multiple onChange={onFilesChange} />
          <button style={styles.button} onClick={uploadFiles}>Ingest Files</button>
        </div>
        <div style={{marginTop:8}}><b>Selected:</b> {filesPreview.join(", ") || "none"}</div>

        <div style={{marginTop:12}}>
          <label style={{marginRight:8}}>
            <input type="checkbox" checked={csvRowMode} onChange={() => setCsvRowMode(s=>!s)} />
            {" "}CSV row-mode (rows indexed as separate chunks)
          </label>
          <small style={{display:"block", color:"#666"}}>CSV ingestion uses row-mode by default (backend also enforces this).</small>
        </div>

        <div style={{marginTop:14}}>
          <label className="label">Ingest URLs (one per line)</label>
          <textarea rows={4} style={styles.textarea} value={urlsText} onChange={(e)=>setUrlsText(e.target.value)} placeholder="https://example.com/article1"></textarea>
          <button style={styles.button} onClick={ingestUrls}>Ingest URLs</button>
        </div>

        <div style={{marginTop:12, color:"#0b63ff"}}>{uploadStatus}</div>

        <div style={{marginTop:14}}>
          <button style={styles.secondary} onClick={fetchMeta}>Show latest indexed rows</button>
        </div>

        {metaList.length > 0 && (
          <div style={{marginTop:12}}>
            <h4>Latest indexed items (most recent first)</h4>
            {metaList.map((m,i) => (
              <div key={i} style={{padding:8, borderBottom:"1px solid #eee"}}>
                <div style={{fontSize:13, color:"#444"}}><b>{m.chunk_id}</b> — {m.title || m.url || "no title"}</div>
                <div style={{marginTop:6, fontSize:14}}>{m.text}</div>
              </div>
            ))}
          </div>
        )}

      </div>

      <div style={styles.card}>
        <h3>Ask a question</h3>
        <textarea rows={4} style={styles.textarea} value={query} onChange={(e)=>setQuery(e.target.value)} placeholder="e.g., What movies are in movies.csv?" />
        <div style={{display:"flex", gap:10, marginTop:10}}>
          <button style={styles.button} onClick={ask}>Search</button>
          <button style={styles.secondary} onClick={()=>{ setQuery("Summarize the contents of the uploaded CSV files."); ask(); }}>Quick summarize</button>
        </div>

        <div style={{marginTop:14}}>
          {results.map((r,i) =>
            r.loading ? <p key={i}>Searching...</p> :
            r.error ? <div key={i} style={styles.error}>Error: {r.error}</div> :
            <div key={i} style={styles.resultBox}>
              <div style={{fontSize:13}}><strong>Score:</strong> {r.score?.toFixed(4)}</div>
              <div style={{fontSize:13, color:"#666"}}><strong>Source:</strong> {r.title || r.url || r.source || "unknown"}</div>
              <div style={{marginTop:8}}>{r.text}</div>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}

const styles = {
  container: { fontFamily: "Inter, Arial, sans-serif", padding: 20, background: "#f4f6f9", minHeight: "100vh" },
  title: { textAlign: "center", marginBottom: 12 },
  card: { background: "#fff", margin: "20px auto", padding: 16, maxWidth: 900, borderRadius: 10, boxShadow: "0 6px 18px rgba(0,0,0,0.08)" },
  button: { padding: "10px 14px", background: "#0066ff", color: "#fff", border: "none", borderRadius: 8, cursor: "pointer" },
  secondary: { padding: "8px 12px", background: "#eee", color: "#222", border: "none", borderRadius: 8, cursor: "pointer" },
  textarea: { width: "100%", padding: 10, borderRadius: 8, border: "1px solid #ddd" },
  resultBox: { marginTop: 12, padding: 12, background: "#fff", borderLeft: "4px solid #0066ff", borderRadius: 6 },
  error: { color: "red" }
};
