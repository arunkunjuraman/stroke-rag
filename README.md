# 🧠 Stroke-RAG Clinical Assistant

**Intelligent Retrieval-Augmented Guidance for Clinical Stroke Protocols.**

Stroke-RAG is a clinical decision-support system designed to provide precise, evidence-based answers to complex stroke guideline queries. By leveraging advanced Retrieval-Augmented Generation (RAG) architectures, it ensures high faithfulness to medical literature while maintaining rapid response times through semantic caching.

---

## 🚀 Key Features

### 🔍 Advanced Retrieval Engine
- **Hybrid Search**: Combines the precision of **Parent Document Retrieval** (vector search) with the robust keyword matching of **BM25**.
- **Multi-Query Expansion**: Generates multiple clinical search variations for every question to ensure maximum recall.
- **Cohere v3.0 Reranking**: Re-orders retrieved candidates using a state-of-the-art cross-encoder to prioritize the most relevant treatment protocols.
- **Contextual Ingestion**: Automatically enriches document chunks with guideline titles and metadata to prevent loss of context during retrieval.

### ⚡ Hybrid Semantic Caching
- **Semantic Mapping**: Uses a dedicated vector index to match questions by intent, not just exact wording.
- **Reliable Fallback**: A multi-tiered cache that prioritize **Redis** for speed but automatically falls back to **Local JSON Storage** to ensure zero downtime.
- **Efficient UI**: Visual feedback in the interface indicates when a response has been lightning-fast served from the cache.

### 📊 Automated Clinical Evaluation
- **RAGAS Integration**: Built-in evaluation suite for measuring performance across key metrics:
  - **Faithfulness**: Ensure answers accurately reflect the guidelines.
  - **Recall & Precision**: Validate that the right medical evidence is retrieved.
  - **Answer Relevancy**: Guarantee that responses directly address clinical needs.
- **Stable Execution**: Serialized evaluation runs provide stability and reliability during performance benchmarking.

### 🎨 User Interface
- **Clean Clinical Design**: A custom-styled Streamlit interface tailored for readability.
- **Evidence Traceability**: Every answer includes "Supporting Evidence" expanders showing the exact **File** and **Page Number** from the source guidelines.

---

## 🛠️ Technology Stack
- **Core Orchestration**: `LangGraph` & `LangChain`
- **Large Language Model**: `GPT-4o`
- **Embeddings**: `OpenAI text-embedding-3-large`
- **Re-ranker**: `Cohere rerank-english-v3.0`
- **Vector Database**: `ChromaDB`
- **Caching**: `Redis` & `Local JSON`
- **Frontend**: `Streamlit`
- **Evaluation**: `RAGAS`

---

## 📂 Project Structure
```text
stroke-rag/
├── src/
│   ├── app.py           # Streamlit Chat Interface
│   ├── brain.py         # RAG Graph Logic (LangGraph)
│   ├── cache.py         # Hybrid Semantic Cache
│   ├── ingestion.py     # Guideline Indexing Pipeline
│   └── evaluator.py     # RAGAS Scoring Script
├── data/
│   └── guidelines/      # PDF Guideline Repository
├── prompts/             # Versioned Clinical Prompt Templates
├── tests/               # Golden Evaluation Datasets
└── requirements.txt     # Project Dependencies
```

---

## ⚙️ Setup & Installation

### Prerequisite: WSL (Recommended)
This project is optimized for execution within **Windows Subsystem for Linux (WSL)**.

1. **Clone the Repository**
   ```bash
   git clone https://github.com/arunkunjuraman/stroke-rag.git
   cd stroke-rag
   ```

2. **Setup Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_key
   COHERE_API_KEY=your_cohere_key
   REDIS_URL=redis://localhost:6379  # Optional: defaults to local file cache
   ```

---

## 📖 Usage

### Step 1: Ingest Guidelines
Place your guideline PDFs in `data/guidelines/` and run the ingestion pipeline:
```bash
python src/ingestion.py
```

### Step 2: Start the Assistant
Launch the Streamlit interface:
```bash
streamlit run src/app.py
```

### Step 3: Run Evaluation
To benchmark performance against the clinical evaluation set:
```bash
python src/evaluator.py
```


---

## 📚 Data Sources

The assistant is grounded in the following clinical guidelines:

- **2024 AHA/ASA Primary Prevention of Stroke Guideline**
  [Download PDF](https://www.ahajournals.org/doi/pdf/10.1161/STR.0000000000000475)
- **2021 Stroke Prevention Guidelines**
  [Download PDF](https://www.ahajournals.org/doi/pdf/10.1161/STR.0000000000000375)
- **2021 Chest Pain / Heart Attack Diagnosis**
  [Download PDF](https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000001029)
- **2023 AHA/ACC/ACCP/ASPC/NLA/PCNA Management of Chronic Coronary Disease**
  [Access Article](https://www.ahajournals.org/doi/epub/10.1161/CIR.0000000000001168)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
