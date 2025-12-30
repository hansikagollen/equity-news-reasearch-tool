# 📊 Equity News Research Assistant

A **real-time document and news research platform** that enables semantic search and analysis over
PDFs, CSV datasets, text files, and live news articles using **vector search**.

The application is designed for **financial research, data exploration, and news analysis**, and demonstrates how modern retrieval systems used in LLM pipelines are built in practice.

---

## 🚀 What This Project Does

- Ingests **PDF, CSV, TXT files**, and **live news URLs**
- Converts documents into **semantic embeddings**
- Indexes data using **FAISS vector search**
- Allows users to **ask natural language questions**
- Displays **real-time ingestion progress** via WebSockets
- Provides fast, accurate **semantic search results**

---

## 🧠 Why This Project Matters

Analyzing large datasets and financial news manually is slow and inefficient.
This project shows how **vector databases and semantic search** can be used to build scalable,
real-time research tools similar to those used in modern AI-powered products.

---

## 🏗️ High-Level Architecture

**Flow:**

# 📊 Equity News Research Assistant

A **real-time document and news research platform** that enables semantic search and analysis over
PDFs, CSV datasets, text files, and live news articles using **vector search**.

The application is designed for **financial research, data exploration, and news analysis**, and demonstrates how modern retrieval systems used in LLM pipelines are built in practice.

---

## 🚀 What This Project Does

- Ingests **PDF, CSV, TXT files**, and **live news URLs**
- Converts documents into **semantic embeddings**
- Indexes data using **FAISS vector search**
- Allows users to **ask natural language questions**
- Displays **real-time ingestion progress** via WebSockets
- Provides fast, accurate **semantic search results**

---

## 🧠 Why This Project Matters

Analyzing large datasets and financial news manually is slow and inefficient.
This project shows how **vector databases and semantic search** can be used to build scalable,
real-time research tools similar to those used in modern AI-powered products.

---

## 🏗️ High-Level Architecture

**Flow:**

# 📊 Equity News Research Assistant

A **real-time document and news research platform** that enables semantic search and analysis over
PDFs, CSV datasets, text files, and live news articles using **vector search**.

The application is designed for **financial research, data exploration, and news analysis**, and demonstrates how modern retrieval systems used in LLM pipelines are built in practice.

---

## 🚀 What This Project Does

- Ingests **PDF, CSV, TXT files**, and **live news URLs**
- Converts documents into **semantic embeddings**
- Indexes data using **FAISS vector search**
- Allows users to **ask natural language questions**
- Displays **real-time ingestion progress** via WebSockets
- Provides fast, accurate **semantic search results**

---

## 🧠 Why This Project Matters

Analyzing large datasets and financial news manually is slow and inefficient.
This project shows how **vector databases and semantic search** can be used to build scalable,
real-time research tools similar to those used in modern AI-powered products.

---

## 🏗️ High-Level Architecture

**Flow:**

Upload / URL
↓
Text Extraction & Chunking
↓
Sentence Embeddings (MiniLM)
↓
FAISS Vector Index
↓
Semantic Search Query
↓
Relevant Results Returned to UI


**Real-time updates** are streamed to the frontend using **WebSockets**.

---

## 🧩 Tech Stack

### Backend
- **FastAPI** – API server
- **FAISS** – Vector similarity search
- **SentenceTransformers (MiniLM)** – Text embeddings
- **Trafilatura** – Web article extraction
- **PyPDF** – PDF parsing
- **WebSockets** – Real-time progress updates

### Frontend
- **React**
- **WebSockets**
- **Fetch API**
- Custom UI for ingestion, progress, and search

---

## 📂 Supported Inputs

- 📄 PDF reports  
- 📊 CSV datasets (row-wise indexing)  
- 📝 Text files  
- 🌐 Live news articles via URLs  

---

## 🔍 Example Use Cases

- “Summarize the uploaded retail sales dataset”
- “What does the BBC article discuss?”
- “What trends appear in the uploaded CSV?”
- “Search for information related to Nvidia in the uploaded news”

---

## 🖥️ Running the Application

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend:app --host 127.0.0.1 --port 8000


Frontend
cd frontend
npm install
npm run dev



