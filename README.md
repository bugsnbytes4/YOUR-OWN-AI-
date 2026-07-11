# VectorDB (Python) — Build a Vector Database from Scratch

A fully working vector database built from scratch in Python, with a web UI.
Implements **HNSW**, **KD-Tree**, and **Brute Force** search side-by-side, plus
a RAG pipeline powered by a local LLM via **Ollama**.

## What This Project Does

| Feature | Description |
|---|---|
| 3 Search Algorithms | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| 3 Distance Metrics | Cosine similarity, Euclidean distance, Manhattan distance |
| 16D Demo Vectors | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| 2D PCA Scatter Plot | Live visualization of semantic space — watch clusters form |
| Real Document Embedding | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| RAG Pipeline | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| Full REST API | insert, delete, search, benchmark, hnsw-info, and RAG endpoints |

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)                ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

HNSW (Hierarchical Navigable Small World) is the same algorithm used by
Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where
each layer is progressively sparser — searches start at the top layer and
zoom in, achieving roughly O(log N) complexity instead of O(N) for brute
force.

## Prerequisites

You need 2 things installed:

1. **Python 3.9+**
2. **Ollama** (runs the local AI models)

No C++ compiler, no MSYS2 — this port is pure Python (`numpy` for vector
math, `flask` for the web server, `requests` to talk to Ollama).

## Step-by-Step Setup

### Step 1 — Install Ollama

1. Go to https://ollama.com and download it for your OS.
2. Run the installer. Ollama starts automatically (system tray on Windows,
   background service on macOS/Linux).
3. Pull the two required models:

   ```bash
   ollama pull nomic-embed-text   # ~274 MB — embedding model
   ollama pull llama3.2           # ~2 GB   — language model
   ```

4. Verify:

   ```bash
   ollama list
   ```

   You should see both models listed. Minimum specs: 8GB RAM recommended,
   the models use ~3GB total.

### Step 2 — Get the Project

```bash
git clone https://github.com/YOUR_USERNAME/VectorDB.git
cd VectorDB/vectordb-python
```

### Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

(If you hit an "externally managed environment" error on Linux, use
`pip install -r requirements.txt --break-system-packages` or create a venv:
`python -m venv venv && source venv/bin/activate` first.)

### Step 4 — Run Everything

**Terminal 1 — Start Ollama** (if not already running):

```bash
ollama serve
```

**Terminal 2 — Start the VectorDB server**:

```bash
python main.py
```

You should see:

```
=== VectorDB Engine (Python) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open your browser to **http://localhost:8080**.

## Using the Application

### Tab 1: Search (Demo Vectors)

- Type any concept in the search box: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose your algorithm: HNSW, KD-Tree, or Brute Force
- Choose distance metric: Cosine, Euclidean, or Manhattan
- Click **⚡ Search** — results appear with distances, and matches glow on the scatter plot
- Click **▶ Compare All Algos** to run all 3 algorithms and compare their speed

The scatter plot shows all 20 vectors projected to 2D using PCA. The 4
semantic categories (CS, Math, Food, Sports) form distinct clusters — this is
what "semantic similarity" looks like visually.

### Tab 2: Documents (Real Embeddings)

This uses Ollama to generate real 768-dimensional embeddings from any text.

1. Type a title (e.g. `Operating Systems Notes`)
2. Paste any text — lecture notes, textbook paragraphs, articles
3. Click **⚡ Embed & Insert**

Long documents are automatically split into overlapping 250-word chunks.
Each chunk gets its own embedding and is stored in the HNSW index.

### Tab 3: Ask AI (RAG Pipeline)

1. Make sure you've inserted some documents in Tab 2 first
2. Type a question about your documents
3. Click **🤖 Ask AI**

Behind the scenes: your question is embedded with `nomic-embed-text`, HNSW
finds the 3 most semantically similar chunks, those chunks are sent as
context to `llama3.2`, and it generates an answer based only on your
documents. Click a context chip to see exactly which chunk the AI used.

## REST API Reference

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search by raw vector |
| GET | `/search_by_label?q_label=binary+tree&k=5&metric=cosine&algo=hnsw` | K-NN search by demo item label (used by the UI) |
| POST | `/insert` | Insert a demo vector — body `{"vector":[...], "label":"...", "category":"..."}` |
| DELETE | `/delete/:id` | Delete by ID |
| GET | `/items` | List all demo vectors with 2D PCA coordinates |
| GET | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| GET | `/hnsw-info` | HNSW graph structure and layer stats |
| GET | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| GET | `/doc/list` | — | List all stored documents/chunks |
| DELETE | `/doc/delete/:id` | — | Delete a document chunk |
| POST | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| GET | `/status` | — | Ollama status and model info |

### Example: Search via curl

```bash
curl "http://localhost:8080/search_by_label?q_label=binary%20tree&k=3&metric=cosine&algo=hnsw"
```

### Example: Ask a question via curl

```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

## Project Structure

```
vectordb-python/
├── main.py           ← Python backend (HNSW, KD-Tree, BruteForce, Flask REST API, RAG)
├── index.html        ← Frontend (PCA scatter plot, search UI, RAG chat)
├── requirements.txt  ← Python dependencies
└── README.md          ← This file
```

## Architecture (main.py)

```
BruteForce          O(N·d)      Exact, baseline
KDTree               O(log N)    Exact, axis-aligned partitioning
HNSW                 O(log N)    Approximate, multilayer small-world graph

VectorDB             Unified interface over all 3 (16D demo vectors)
DocumentDB           HNSW-only index for real Ollama embeddings (768D)
OllamaClient         HTTP client → /api/embeddings + /api/generate
```

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

Nodes are inserted into a multilayer graph. Each node randomly gets assigned
a maximum layer (exponentially fewer nodes survive at higher layers). Layer 0
has all nodes with many connections; higher layers have far fewer nodes with
longer-range connections.

**Insert**: Start at the top layer, greedily find the nearest node, drop a
layer, repeat. At each layer from the node's assigned max down to 0, run a
beam search (`ef_construction=200`) and connect to the `M` nearest neighbors
bidirectionally.

**Search**: Same greedy descent from the top layer. At layer 0, expand to
`ef` nearest candidates using a priority queue.

Why it's fast: the upper layers act like a highway — you quickly get to the
right neighborhood, then zoom in at layer 0.

### KD-Tree (K-Dimensional Tree)

Binary space partitioning. Each node splits space along one dimension
(cycling through all dimensions). Search prunes entire subtrees when the
closest possible point in that subtree can't beat the current best — the
"ball within hyperslab" check.

**Weakness**: degrades with high dimensions (curse of dimensionality). Works
well for ≤20D, becomes close to brute force at 768D.

### Why HNSW Wins at High Dimensions

KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions,
almost all the space is near the boundary of the hypersphere — few subtrees
get pruned. HNSW's graph-based approach doesn't have this problem, which is
why the document/RAG index (768D) uses HNSW exclusively.

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in header | Run `ollama serve` in a terminal |
| Embedding takes forever | Ollama is downloading the model on first use — wait a couple minutes |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port 8080 already in use | Kill the process using it, or change the port at the bottom of `main.py` |
| LLM answer is slow | Normal — `llama3.2` takes 10–30s on a laptop CPU. Use `llama3.2:1b` for faster answers |

## Use a Smaller/Faster LLM

If `llama3.2` is too slow on your machine, switch to the 1B model:

```bash
ollama pull llama3.2:1b
```

Then edit the top of `main.py`:

```python
GEN_MODEL = "llama3.2:1b"   # change this
```

Restart `python main.py`.
