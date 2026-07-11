"""
VectorDB (Python port) — a from-scratch vector database with HNSW, KD-Tree,
and Brute Force search, plus a RAG pipeline powered by a local LLM via Ollama.

Run:
    pip install -r requirements.txt
    python main.py
Then open http://localhost:8080
"""

import heapq
import math
import random
import re
import time
import uuid

import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2"
DEMO_DIM = 16
CHUNK_WORDS = 250
CHUNK_OVERLAP = 50

app = Flask(__name__, static_folder=None)

# --------------------------------------------------------------------------
# Distance metrics
# --------------------------------------------------------------------------
def cosine_distance(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))


def manhattan_distance(a, b):
    return float(np.sum(np.abs(a - b)))


METRICS = {
    "cosine": cosine_distance,
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
}


# --------------------------------------------------------------------------
# Brute Force  — O(N * d), exact, baseline
# --------------------------------------------------------------------------
class BruteForce:
    def __init__(self, vectors: dict):
        self.vectors = vectors  # id -> np.array

    def search(self, query, k, metric):
        dist_fn = METRICS[metric]
        scored = [(dist_fn(query, v), vid) for vid, v in self.vectors.items()]
        scored.sort(key=lambda x: x[0])
        return scored[:k]


# --------------------------------------------------------------------------
# KD-Tree — O(log N) average, exact, axis-aligned partitioning
# Degrades toward brute force in high dimensions (curse of dimensionality).
# --------------------------------------------------------------------------
class KDNode:
    __slots__ = ("id", "point", "axis", "left", "right")

    def __init__(self, id_, point, axis):
        self.id = id_
        self.point = point
        self.axis = axis
        self.left = None
        self.right = None


class KDTree:
    def __init__(self, vectors: dict):
        self.vectors = vectors
        items = list(vectors.items())
        self.dim = len(items[0][1]) if items else 0
        self.root = self._build(items, 0)

    def _build(self, items, depth):
        if not items:
            return None
        axis = depth % self.dim
        items = sorted(items, key=lambda kv: kv[1][axis])
        mid = len(items) // 2
        node = KDNode(items[mid][0], items[mid][1], axis)
        node.left = self._build(items[:mid], depth + 1)
        node.right = self._build(items[mid + 1 :], depth + 1)
        return node

    def search(self, query, k, metric):
        dist_fn = METRICS[metric]
        best = []  # list of (dist, id), kept sorted, size <= k

        def visit(node):
            if node is None:
                return
            d = dist_fn(query, node.point)
            if len(best) < k:
                best.append((d, node.id))
                best.sort(key=lambda x: x[0])
            elif d < best[-1][0]:
                best.append((d, node.id))
                best.sort(key=lambda x: x[0])
                best.pop()

            axis = node.axis
            diff = query[axis] - node.point[axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            visit(near)
            # Pruning: skip far subtree if it cannot possibly contain a closer
            # point than our current worst candidate. This bound is exact for
            # euclidean/manhattan; treated as a heuristic for cosine.
            if len(best) < k or abs(diff) < best[-1][0]:
                visit(far)

        visit(self.root)
        return best


# --------------------------------------------------------------------------
# HNSW — Hierarchical Navigable Small World graph
# O(log N) approximate search, the algorithm behind Pinecone/Weaviate/Chroma.
# --------------------------------------------------------------------------
class HNSW:
    def __init__(self, M=16, ef_construction=200, metric="cosine", seed=42):
        self.M = M
        self.Mmax = M
        self.Mmax0 = M * 2
        self.ef_construction = ef_construction
        self.metric = metric
        self.mL = 1 / math.log(M)
        self.enter_point = None
        self.max_level = -1
        self.levels = {}  # id -> level
        self.neighbors = {}  # id -> {layer: set(ids)}
        self.vectors = {}  # id -> np.array
        self.rng = random.Random(seed)

    def _dist(self, a, b):
        return METRICS[self.metric](a, b)

    def _random_level(self):
        return int(-math.log(self.rng.random() + 1e-12) * self.mL)

    def insert(self, id_, vector):
        self.vectors[id_] = vector
        level = self._random_level()
        self.levels[id_] = level
        self.neighbors[id_] = {l: set() for l in range(level + 1)}

        if self.enter_point is None:
            self.enter_point = id_
            self.max_level = level
            return

        ep = self.enter_point
        for lc in range(self.max_level, level, -1):
            res = self._search_layer(vector, ep, 1, lc)
            if res:
                ep = res[0][1]

        for lc in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(vector, ep, self.ef_construction, lc)
            selected = sorted(candidates, key=lambda x: x[0])[: self.M]
            for d, nid in selected:
                self.neighbors[id_][lc].add(nid)
                self.neighbors[nid].setdefault(lc, set()).add(id_)
                maxconn = self.Mmax0 if lc == 0 else self.Mmax
                if len(self.neighbors[nid][lc]) > maxconn:
                    nbrs = list(self.neighbors[nid][lc])
                    nbrs.sort(key=lambda x: self._dist(self.vectors[nid], self.vectors[x]))
                    self.neighbors[nid][lc] = set(nbrs[:maxconn])
            if selected:
                ep = selected[0][1]

        if level > self.max_level:
            self.max_level = level
            self.enter_point = id_

    def _search_layer(self, query, entry, ef, layer):
        visited = {entry}
        d0 = self._dist(query, self.vectors[entry])
        candidates = [(d0, entry)]
        result = [(d0, entry)]
        heapq.heapify(candidates)

        while candidates:
            d, c = heapq.heappop(candidates)
            worst = max(result, key=lambda x: x[0])[0] if result else float("inf")
            if d > worst and len(result) >= ef:
                break
            for nb in self.neighbors.get(c, {}).get(layer, []):
                if nb not in visited:
                    visited.add(nb)
                    dn = self._dist(query, self.vectors[nb])
                    worst = max(result, key=lambda x: x[0])[0] if result else float("inf")
                    if len(result) < ef or dn < worst:
                        heapq.heappush(candidates, (dn, nb))
                        result.append((dn, nb))
                        result.sort(key=lambda x: x[0])
                        if len(result) > ef:
                            result.pop()
        return result

    def search(self, query, k, ef=50):
        if self.enter_point is None:
            return []
        ep = self.enter_point
        for lc in range(self.max_level, 0, -1):
            res = self._search_layer(query, ep, 1, lc)
            if res:
                ep = res[0][1]
        candidates = self._search_layer(query, ep, max(ef, k), 0)
        candidates.sort(key=lambda x: x[0])
        return candidates[:k]

    def info(self):
        layer_counts = {}
        for id_, lvl in self.levels.items():
            for l in range(lvl + 1):
                layer_counts[l] = layer_counts.get(l, 0) + 1
        return {
            "num_nodes": len(self.vectors),
            "max_level": self.max_level,
            "layer_counts": layer_counts,
            "M": self.M,
            "ef_construction": self.ef_construction,
        }


# --------------------------------------------------------------------------
# Demo dataset — 20 vectors, 16D, across 4 semantic categories
# --------------------------------------------------------------------------
CATEGORY_LABELS = {
    "CS": ["binary tree", "linked list", "hash map", "recursion", "binary search"],
    "Math": ["calculus", "linear algebra", "probability", "integral", "derivative"],
    "Food": ["sushi", "pizza", "pasta", "curry", "tacos"],
    "Sports": ["basketball", "soccer", "tennis", "swimming", "baseball"],
}


def build_demo_vectors():
    rng = np.random.default_rng(7)
    categories = list(CATEGORY_LABELS.keys())
    # QR decomposition gives mutually orthogonal centroids, guaranteeing the
    # four semantic categories separate cleanly regardless of random seed.
    random_mat = rng.normal(0, 1, (DEMO_DIM, len(categories)))
    q, _ = np.linalg.qr(random_mat)
    vectors, meta = {}, {}
    for idx, cat in enumerate(categories):
        centroid = q[:, idx]
        for label in CATEGORY_LABELS[cat]:
            vid = str(uuid.uuid4())[:8]
            noise = rng.normal(0, 0.15, DEMO_DIM)
            vec = centroid + noise
            vectors[vid] = vec
            meta[vid] = {"label": label, "category": cat}
    return vectors, meta


# --------------------------------------------------------------------------
# VectorDB — unified interface over BruteForce / KD-Tree / HNSW (16D demo)
# --------------------------------------------------------------------------
class VectorDB:
    def __init__(self):
        self.vectors, self.meta = build_demo_vectors()
        self._rebuild_all()

    def _rebuild_all(self):
        self.bruteforce = BruteForce(dict(self.vectors))
        self.kdtree = KDTree(dict(self.vectors)) if self.vectors else None
        hnsw = HNSW(metric="cosine")
        for vid, v in self.vectors.items():
            hnsw.insert(vid, v)
        self.hnsw = hnsw

    def insert(self, vector, label="untitled", category="custom"):
        vid = str(uuid.uuid4())[:8]
        self.vectors[vid] = np.array(vector, dtype=float)
        self.meta[vid] = {"label": label, "category": category}
        self._rebuild_all()
        return vid

    def delete(self, vid):
        if vid in self.vectors:
            del self.vectors[vid]
            del self.meta[vid]
            self._rebuild_all()
            return True
        return False

    def search(self, query, k, metric, algo):
        query = np.array(query, dtype=float)
        if algo == "bruteforce":
            results = self.bruteforce.search(query, k, metric)
        elif algo == "kdtree":
            results = self.kdtree.search(query, k, metric) if self.kdtree else []
        elif algo == "hnsw":
            if metric != self.hnsw.metric:
                # Rebuild the graph if a different metric is requested.
                hnsw = HNSW(metric=metric)
                for vid, v in self.vectors.items():
                    hnsw.insert(vid, v)
                self.hnsw = hnsw
            results = self.hnsw.search(query, k)
        else:
            raise ValueError(f"Unknown algo: {algo}")
        return [
            {"id": vid, "distance": d, **self.meta.get(vid, {})} for d, vid in results
        ]

    def benchmark(self, query, k, metric):
        out = {}
        for algo in ("bruteforce", "kdtree", "hnsw"):
            t0 = time.perf_counter()
            res = self.search(query, k, metric, algo)
            t1 = time.perf_counter()
            out[algo] = {"results": res, "time_ms": (t1 - t0) * 1000}
        return out

    def pca_2d(self):
        if not self.vectors:
            return {}
        ids = list(self.vectors.keys())
        mat = np.array([self.vectors[i] for i in ids])
        mat_centered = mat - mat.mean(axis=0)
        cov = np.cov(mat_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        top2 = eigvecs[:, order[:2]]
        proj = mat_centered @ top2
        return {ids[i]: (float(proj[i, 0]), float(proj[i, 1])) for i in range(len(ids))}


# --------------------------------------------------------------------------
# DocumentDB — HNSW-only index for real Ollama embeddings (768D)
# --------------------------------------------------------------------------
class DocumentDB:
    def __init__(self):
        self.hnsw = HNSW(metric="cosine")
        self.chunks = {}  # id -> {title, text, doc_id}

    def chunk_text(self, text):
        words = text.split()
        if not words:
            return []
        step = CHUNK_WORDS - CHUNK_OVERLAP
        chunks = []
        for start in range(0, len(words), step):
            piece = words[start : start + CHUNK_WORDS]
            if piece:
                chunks.append(" ".join(piece))
            if start + CHUNK_WORDS >= len(words):
                break
        return chunks

    def insert_document(self, title, text, ollama):
        doc_id = str(uuid.uuid4())[:8]
        chunk_texts = self.chunk_text(text)
        chunk_ids = []
        for chunk in chunk_texts:
            vec = ollama.embed(chunk)
            cid = str(uuid.uuid4())[:8]
            self.hnsw.insert(cid, np.array(vec, dtype=float))
            self.chunks[cid] = {"title": title, "text": chunk, "doc_id": doc_id}
            chunk_ids.append(cid)
        return doc_id, chunk_ids

    def delete_chunk(self, cid):
        return self.chunks.pop(cid, None) is not None

    def search(self, query_vec, k=3):
        results = self.hnsw.search(np.array(query_vec, dtype=float), k)
        out = []
        for d, cid in results:
            if cid in self.chunks:
                out.append({"id": cid, "distance": d, **self.chunks[cid]})
        return out

    def list_documents(self):
        return [{"id": cid, **meta} for cid, meta in self.chunks.items()]


# --------------------------------------------------------------------------
# OllamaClient — talks to a local Ollama instance
# --------------------------------------------------------------------------
class OllamaClient:
    def __init__(self, base_url=OLLAMA_URL):
        self.base_url = base_url

    def is_online(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException:
            return []

    def embed(self, text):
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["embedding"]
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url} ({e})") from e

    def generate(self, prompt, model=GEN_MODEL):
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url} ({e})") from e


# --------------------------------------------------------------------------
# App state
# --------------------------------------------------------------------------
db = VectorDB()
docdb = DocumentDB()
ollama = OllamaClient()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def parse_vector(param):
    return [float(x) for x in param.split(",") if x.strip() != ""]


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# --------------------------------------------------------------------------
# Demo vector endpoints
# --------------------------------------------------------------------------
@app.route("/search", methods=["GET"])
def search():
    v = parse_vector(request.args.get("v", ""))
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    algo = request.args.get("algo", "hnsw")
    t0 = time.perf_counter()
    results = db.search(v, k, metric, algo)
    t1 = time.perf_counter()
    return jsonify({"results": results, "time_ms": (t1 - t0) * 1000, "algo": algo, "metric": metric})


@app.route("/insert", methods=["POST"])
def insert():
    body = request.get_json(force=True)
    vid = db.insert(body["vector"], body.get("label", "untitled"), body.get("category", "custom"))
    return jsonify({"id": vid})


@app.route("/delete/<vid>", methods=["DELETE"])
def delete(vid):
    ok = db.delete(vid)
    return jsonify({"deleted": ok})


@app.route("/items", methods=["GET"])
def items():
    coords = db.pca_2d()
    out = []
    for vid, meta in db.meta.items():
        x, y = coords.get(vid, (0.0, 0.0))
        out.append({"id": vid, "x": x, "y": y, **meta})
    return jsonify(out)


@app.route("/benchmark", methods=["GET"])
def benchmark():
    v = parse_vector(request.args.get("v", ""))
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    return jsonify(db.benchmark(v, k, metric))


def _find_by_label(label):
    label = label.strip().lower()
    for vid, meta in db.meta.items():
        if meta["label"].lower() == label:
            return vid
    return None


@app.route("/search_by_label", methods=["GET"])
def search_by_label():
    # Convenience wrapper the UI uses: look up a demo vector by its label
    # (e.g. "binary tree") instead of requiring raw 16-dimensional floats.
    vid = _find_by_label(request.args.get("q_label", ""))
    if vid is None:
        return jsonify({"error": "no demo vector with that label"}), 404
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    algo = request.args.get("algo", "hnsw")
    t0 = time.perf_counter()
    results = db.search(db.vectors[vid], k, metric, algo)
    t1 = time.perf_counter()
    return jsonify({"results": results, "time_ms": (t1 - t0) * 1000, "algo": algo, "metric": metric})


@app.route("/benchmark_by_label", methods=["GET"])
def benchmark_by_label():
    vid = _find_by_label(request.args.get("q_label", ""))
    if vid is None:
        return jsonify({"error": "no demo vector with that label"}), 404
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    return jsonify(db.benchmark(db.vectors[vid], k, metric))


@app.route("/hnsw-info", methods=["GET"])
def hnsw_info():
    return jsonify(db.hnsw.info())


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(
        {
            "num_vectors": len(db.vectors),
            "dims": DEMO_DIM,
            "categories": list(CATEGORY_LABELS.keys()),
            "num_documents": len(set(c["doc_id"] for c in docdb.chunks.values())),
            "num_chunks": len(docdb.chunks),
        }
    )


# --------------------------------------------------------------------------
# Document & RAG endpoints
# --------------------------------------------------------------------------
@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    body = request.get_json(force=True)
    title, text = body.get("title", "untitled"), body.get("text", "")
    if not text.strip():
        return jsonify({"error": "text is required"}), 400
    try:
        doc_id, chunk_ids = docdb.insert_document(title, text, ollama)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    return jsonify({"doc_id": doc_id, "chunk_ids": chunk_ids, "num_chunks": len(chunk_ids)})


@app.route("/doc/list", methods=["GET"])
def doc_list():
    return jsonify(docdb.list_documents())


@app.route("/doc/delete/<cid>", methods=["DELETE"])
def doc_delete(cid):
    return jsonify({"deleted": docdb.delete_chunk(cid)})


@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    body = request.get_json(force=True)
    question = body.get("question", "")
    k = int(body.get("k", 3))
    if not question.strip():
        return jsonify({"error": "question is required"}), 400

    try:
        q_vec = ollama.embed(question)
        context_chunks = docdb.search(q_vec, k)
        context_text = "\n\n".join(f"[{c['title']}]: {c['text']}" for c in context_chunks)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer isn't in the context, say so.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        )
        answer = ollama.generate(prompt)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    return jsonify({"answer": answer, "context": context_chunks})


@app.route("/status", methods=["GET"])
def status():
    online = ollama.is_online()
    return jsonify(
        {
            "ollama_online": online,
            "models": ollama.list_models() if online else [],
            "embed_model": EMBED_MODEL,
            "gen_model": GEN_MODEL,
        }
    )


# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== VectorDB Engine (Python) ===")
    print("http://localhost:8080")
    print(f"{len(db.vectors)} demo vectors | {DEMO_DIM} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama.is_online() else 'OFFLINE'}")
    print(f"  embed model: {EMBED_MODEL}  gen model: {GEN_MODEL}")
    app.run(host="0.0.0.0", port=8080, debug=False)
