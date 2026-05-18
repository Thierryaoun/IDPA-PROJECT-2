"""
app.py
------
Flask API server for the Project 2 clustering GUI.

Exposes the ClusteringPipeline as REST endpoints consumed by the
single-page frontend in frontend/templates/index.html.

Run:
    python app.py
Then open http://localhost:5000 in a browser.
"""

import json
import os
import sys
import threading
import time
import uuid
from typing import Dict, Any

from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from clustering import (
    ClusteringPipeline,
    ClusterResult,
    DistanceMatrix,
    setup_logging,
)
from clustering.utils import load_json

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=os.path.join(_ROOT, "frontend", "templates"),
    static_folder=os.path.join(_ROOT, "frontend", "static"),
)
CORS(app)

setup_logging(level="INFO", log_file="output/clustering/clustering.log", silent=True)

_OUTPUT_DIR  = os.path.join(_ROOT, "output", "clustering")
_TREE_DIR    = os.path.join(_ROOT, "preprocessed_trees")
_CACHE_FULL  = os.path.join(_OUTPUT_DIR, "distance_matrix.json")

os.makedirs(_OUTPUT_DIR, exist_ok=True)

# Shared pipeline instance (created once, reused across requests)
_pipeline: ClusteringPipeline = ClusteringPipeline(
    tree_dir=_TREE_DIR,
    output_dir=_OUTPUT_DIR,
    cache_matrix=True,
)

# Background job registry  {job_id: {status, progress, message, error, result_file}}
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _job_update(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _all_countries() -> list:
    """Return sorted list of country names available in preprocessed_trees/.

    Excludes diff/patch/meta files (those starting with '_').
    """
    names = []
    if os.path.isdir(_TREE_DIR):
        for fname in os.listdir(_TREE_DIR):
            if fname.endswith(".json") and not fname.startswith("_"):
                names.append(fname[:-5])
    return sorted(names)


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── API: countries ────────────────────────────────────────────────────────────

@app.route("/api/countries")
def api_countries():
    countries = _all_countries()
    return jsonify({"countries": countries, "total": len(countries)})


# ── API: matrix status ────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    has_matrix = _pipeline.distance_matrix is not None
    cached = os.path.exists(_CACHE_FULL)
    n_loaded = _pipeline.n_documents
    return jsonify({
        "matrix_loaded": has_matrix,
        "matrix_cached": cached,
        "n_loaded": n_loaded,
        "n_available": len(_all_countries()),
    })


# ── API: start matrix computation (async) ────────────────────────────────────

@app.route("/api/matrix/start", methods=["POST"])
def api_matrix_start():
    body = request.get_json(silent=True) or {}
    countries  = body.get("countries")   # None = all
    use_semantic = bool(body.get("use_semantic", True))
    force      = bool(body.get("force", False))

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "type": "matrix",
            "status": "running",
            "progress": 0,
            "message": "Initialising…",
            "error": None,
        }

    def _worker():
        try:
            _job_update(job_id, message="Loading trees…", progress=5)
            _pipeline.load_trees(country_filter=countries)
            n = _pipeline.n_documents
            _job_update(job_id, message=f"{n} trees loaded. Computing distances…", progress=15)

            cache = None if force else _CACHE_FULL
            if force and os.path.exists(_CACHE_FULL):
                os.remove(_CACHE_FULL)

            from clustering.distance_matrix import build_distance_matrix
            _pipeline._dm = build_distance_matrix(
                trees=_pipeline.trees,
                use_semantic=use_semantic,
                cache_path=_CACHE_FULL,
            )
            _job_update(job_id, status="done", progress=100,
                        message=f"Distance matrix ready ({n}x{n}).")
        except Exception as exc:
            _job_update(job_id, status="error", error=str(exc),
                        message="Failed.")

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"job_id": job_id})


# ── API: job poll ─────────────────────────────────────────────────────────────

@app.route("/api/job/<job_id>")
def api_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ── API: get loaded distance matrix ──────────────────────────────────────────

def _all_cached_matrices() -> list:
    """Return list of all cached distance matrix JSON files in output_dir."""
    found = []
    if os.path.isdir(_OUTPUT_DIR):
        for fname in sorted(os.listdir(_OUTPUT_DIR)):
            if fname.startswith("distance_matrix") and fname.endswith(".json"):
                found.append(os.path.join(_OUTPUT_DIR, fname))
    return found


@app.route("/api/matrix")
def api_matrix():
    """Return the current distance matrix.

    Optional query param ?n=<int> requests the matrix whose n equals that value.
    Falls back to the first cached matrix if the loaded one does not match.
    """
    n_hint = request.args.get("n", type=int)

    # If in-memory matrix matches the hint (or no hint), return it
    if _pipeline.distance_matrix is not None:
        if n_hint is None or _pipeline.distance_matrix.n == n_hint:
            dm = _pipeline.distance_matrix
            return jsonify({"labels": dm.labels, "matrix": dm.matrix,
                            "stats": dm.stats, "n": dm.n})

    # Try each cached matrix file; prefer one matching n_hint
    candidates = _all_cached_matrices()
    chosen = None
    for path in candidates:
        try:
            data = load_json(path)
            labels = data.get("labels", [])
            if n_hint is None or len(labels) == n_hint:
                chosen = data
                if n_hint is not None:
                    break   # exact match — stop searching
        except Exception:
            continue

    if chosen is None and candidates:
        # No match — return the first available matrix
        try:
            chosen = load_json(candidates[0])
        except Exception:
            pass

    if chosen is None:
        return jsonify({"error": "No distance matrix found. Run /api/matrix/start first."}), 404

    return jsonify({
        "labels": chosen["labels"],
        "matrix": chosen["matrix"],
        "stats":  chosen.get("stats", {}),
        "n":      len(chosen["labels"]),
    })


# ── API: run clustering ───────────────────────────────────────────────────────

@app.route("/api/cluster", methods=["POST"])
def api_cluster():
    body = request.get_json(silent=True) or {}
    algorithm  = body.get("algorithm", "kmedoids")
    n_clusters = int(body.get("n_clusters", 5))
    linkage    = body.get("linkage", "average")
    init_method = body.get("init_method", "heuristic")
    max_iter   = int(body.get("max_iter", 100))
    tag        = body.get("tag", "")

    if _pipeline.distance_matrix is None:
        if os.path.exists(_CACHE_FULL):
            _pipeline._dm = DistanceMatrix.load(_CACHE_FULL)
            _pipeline._trees = {lbl: None for lbl in _pipeline._dm.labels}
        else:
            return jsonify({"error": "Run /api/matrix/start first"}), 400

    try:
        result = _pipeline.run(
            algorithm=algorithm,
            n_clusters=n_clusters,
            evaluate=True,
            linkage=linkage,
            init_method=init_method,
            max_iter=max_iter,
        )
        path = _pipeline.save_results(result, tag=tag)
        fname = os.path.basename(path)
        return jsonify({"result_file": fname, "result": result.to_dict()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── API: list saved results ───────────────────────────────────────────────────

@app.route("/api/results")
def api_results():
    files = []
    if os.path.isdir(_OUTPUT_DIR):
        for fname in sorted(os.listdir(_OUTPUT_DIR)):
            if fname.endswith(".json") and fname not in ("distance_matrix.json",
                                                          "distance_matrix_demo.json"):
                fpath = os.path.join(_OUTPUT_DIR, fname)
                try:
                    data = load_json(fpath)
                    files.append({
                        "filename": fname,
                        "algorithm": data.get("algorithm"),
                        "n_clusters": data.get("n_clusters"),
                        "silhouette_score": data.get("silhouette_score"),
                        "inertia": data.get("inertia"),
                        "n_documents": data.get("metadata", {}).get("n_documents"),
                    })
                except Exception:
                    pass
    return jsonify({"results": files})


# ── API: load one result ──────────────────────────────────────────────────────

@app.route("/api/result/<filename>")
def api_result(filename):
    fpath = os.path.join(_OUTPUT_DIR, filename)
    if not os.path.exists(fpath):
        return jsonify({"error": "File not found"}), 404
    data = load_json(fpath)
    return jsonify(data)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  Project 2 — Clustering GUI")
    print("  http://localhost:5000")
    print("=" * 60)
    print()
    app.run(debug=False, port=5000, threaded=True)
