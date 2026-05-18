"""
app.py
------
Flask API server for the Project 2 clustering GUI.

Key design invariants (post-fix):
  * Subset matrices NEVER overwrite the full-country cache.
    Every unique label-set gets its own deterministic cache file:
      distance_matrix_<n>_<md5[:8]>.json
  * Every ClusterResult carries metadata["matrix_id"] and
    metadata["matrix_labels"] so the frontend can verify it is
    loading exactly the matrix used to produce that result.
  * /api/matrix?id=<matrix_id> looks up the right matrix file by ID.
  * /api/result/<file> returns matrix_id so the frontend can call
    /api/matrix?id=... and verify label alignment before rendering.
"""

import hashlib
import os
import sys
import threading
import uuid
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from clustering import ClusteringPipeline, ClusterResult, DistanceMatrix, setup_logging
from clustering.utils import load_json, save_json

# ── App & logging ─────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=os.path.join(_ROOT, "frontend", "templates"),
    static_folder=os.path.join(_ROOT, "frontend", "static"),
)
from flask_cors import CORS
CORS(app)

_OUTPUT_DIR = os.path.join(_ROOT, "output", "clustering")
_TREE_DIR   = os.path.join(_ROOT, "preprocessed_trees")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

try:
    setup_logging(level="INFO", log_file=os.path.join(_OUTPUT_DIR, "clustering.log"), silent=True)
except Exception:
    setup_logging(level="INFO", silent=True)   # log file locked — fall back to silent

# Shared pipeline (created once)
_pipeline: ClusteringPipeline = ClusteringPipeline(
    tree_dir=_TREE_DIR,
    output_dir=_OUTPUT_DIR,
    cache_matrix=True,
)

# Background jobs  {job_id: {status, progress, message, error}}
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# ── Cache-path helpers ────────────────────────────────────────────────────────

def _matrix_cache_path(labels: List[str], field_filter=None) -> str:
    """Deterministic, collision-free cache filename for any label+field set.

    Both the country selection and the field filter are hashed so different
    combinations always map to separate cache files.
    """
    key = ",".join(sorted(labels))
    if field_filter:
        key += "|" + ",".join(sorted(field_filter))
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return os.path.join(_OUTPUT_DIR, f"distance_matrix_{len(labels)}_{h}.json")


def _matrix_id_from_labels(labels: List[str], field_set=None) -> str:
    key = ",".join(sorted(labels))
    if field_set:
        key += "|" + ",".join(sorted(field_set))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _all_cached_matrix_paths() -> List[str]:
    """All distance_matrix_*.json files in output dir."""
    return sorted(
        os.path.join(_OUTPUT_DIR, f)
        for f in os.listdir(_OUTPUT_DIR)
        if f.startswith("distance_matrix") and f.endswith(".json")
    ) if os.path.isdir(_OUTPUT_DIR) else []


# ── Country helpers ───────────────────────────────────────────────────────────

def _all_country_stems() -> List[str]:
    """Return sorted filename stems (stable IDs) for all real country trees."""
    if not os.path.isdir(_TREE_DIR):
        return []
    return sorted(
        f[:-5]
        for f in os.listdir(_TREE_DIR)
        if f.endswith(".json") and not f.startswith("_")
    )


# ── Job helpers ───────────────────────────────────────────────────────────────

def _job_update(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── API: countries ────────────────────────────────────────────────────────────

@app.route("/api/countries")
def api_countries():
    stems = _all_country_stems()
    return jsonify({"countries": stems, "total": len(stems)})


# ── API: status ───────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    has_matrix  = _pipeline.distance_matrix is not None
    any_cached  = bool(_all_cached_matrix_paths())
    n_loaded    = _pipeline.n_documents
    return jsonify({
        "matrix_loaded": has_matrix,
        "matrix_cached": any_cached,
        "n_loaded":      n_loaded,
        "n_available":   len(_all_country_stems()),
        "matrix_id":     _pipeline.distance_matrix.matrix_id if has_matrix else None,
    })


# ── API: start matrix computation (async) ────────────────────────────────────

@app.route("/api/matrix/start", methods=["POST"])
def api_matrix_start():
    body         = request.get_json(silent=True) or {}
    countries    = body.get("countries")        # None → all
    use_semantic = bool(body.get("use_semantic", True))
    force        = bool(body.get("force", False))
    fields       = body.get("fields")           # None → all fields; list → filter

    field_filter = set(fields) if fields else None

    if field_filter is not None and len(field_filter) == 0:
        return jsonify({"error": "Select at least one field group."}), 400

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "type": "matrix", "status": "running",
            "progress": 0, "message": "Initialising…", "error": None,
        }

    def _worker():
        try:
            _job_update(job_id, message="Loading trees…", progress=5)
            _pipeline.load_trees(country_filter=countries)
            n = _pipeline.n_documents
            _job_update(job_id,
                        message=f"{n} trees loaded. Computing distances…",
                        progress=15)

            loaded_labels = sorted(_pipeline.trees.keys())
            cache_path    = _matrix_cache_path(loaded_labels, field_filter)

            if force and os.path.exists(cache_path):
                os.remove(cache_path)
                _job_update(job_id, message="Cache cleared. Recomputing…", progress=16)

            from clustering.distance_matrix import build_distance_matrix
            _pipeline._dm = build_distance_matrix(
                trees=_pipeline.trees,
                use_semantic=use_semantic,
                cache_path=cache_path,
                field_filter=field_filter,
            )
            _job_update(job_id,
                        status="done", progress=100,
                        message=f"Matrix ready ({n}x{n}).",
                        matrix_id=_pipeline._dm.matrix_id,
                        cache_file=os.path.basename(cache_path))
        except Exception as exc:
            _job_update(job_id, status="error", error=str(exc), message="Failed.")

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


# ── API: get distance matrix ──────────────────────────────────────────────────

@app.route("/api/matrix")
def api_matrix():
    """Return a distance matrix.

    Query params (use exactly one):
      ?id=<matrix_id>   — preferred: look up the matrix that produced a result
      ?n=<int>          — fallback: return the first cached matrix of this size
    No params           — return the currently loaded matrix, or the largest cached one.
    """
    want_id = request.args.get("id")
    want_n  = request.args.get("n", type=int)

    # 1. Try in-memory pipeline matrix
    dm = _pipeline.distance_matrix
    if dm is not None:
        if want_id is None and want_n is None:
            return _dm_response(dm)
        if want_id and dm.matrix_id == want_id:
            return _dm_response(dm)
        if want_n and dm.n == want_n and want_id is None:
            return _dm_response(dm)

    # 2. Search cached files
    for path in _all_cached_matrix_paths():
        try:
            raw = load_json(path)
        except Exception:
            continue
        labels    = raw.get("labels", [])
        field_set = raw.get("selected_fields")
        mid       = raw.get("matrix_id") or _matrix_id_from_labels(labels, field_set)

        base_resp = {"labels": labels, "matrix": raw["matrix"],
                     "stats": raw.get("stats", {}), "n": len(labels),
                     "matrix_id": mid, "selected_fields": field_set}
        if want_id and mid == want_id:
            return jsonify(base_resp)
        if want_n and len(labels) == want_n and want_id is None:
            return jsonify(base_resp)
        if want_id is None and want_n is None:
            return jsonify(base_resp)

    return jsonify({"error": "No matching distance matrix found. Run /api/matrix/start first."}), 404


def _dm_response(dm: DistanceMatrix):
    return jsonify({
        "labels":          dm.labels,
        "matrix":          dm.matrix,
        "stats":           dm.stats,
        "n":               dm.n,
        "matrix_id":       dm.matrix_id,
        "selected_fields": sorted(dm.field_set) if dm.field_set else None,
    })


# ── API: run clustering ───────────────────────────────────────────────────────

@app.route("/api/cluster", methods=["POST"])
def api_cluster():
    body        = request.get_json(silent=True) or {}
    algorithm   = body.get("algorithm", "kmedoids")
    n_clusters  = int(body.get("n_clusters", 5))
    linkage     = body.get("linkage", "average")
    init_method = body.get("init_method", "heuristic")
    max_iter    = int(body.get("max_iter", 100))
    tag         = body.get("tag", "")

    # Ensure a matrix is loaded
    if _pipeline.distance_matrix is None:
        cached = _all_cached_matrix_paths()
        if cached:
            _pipeline._dm = DistanceMatrix.load(cached[-1])
            _pipeline._trees = {lbl: None for lbl in _pipeline._dm.labels}
        else:
            return jsonify({"error": "Run /api/matrix/start first."}), 400

    try:
        result = _pipeline.run(
            algorithm=algorithm,
            n_clusters=n_clusters,
            evaluate=True,
            linkage=linkage,
            init_method=init_method,
            max_iter=max_iter,
        )
        # n_documents was not set by pipeline.run() in some paths — ensure it
        result.metadata.setdefault("n_documents", _pipeline.distance_matrix.n)
        path  = _pipeline.save_results(result, tag=tag)
        fname = os.path.basename(path)
        return jsonify({"result_file": fname, "result": result.to_dict()})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── API: list saved results ───────────────────────────────────────────────────

@app.route("/api/results")
def api_results():
    files = []
    if os.path.isdir(_OUTPUT_DIR):
        for fname in sorted(os.listdir(_OUTPUT_DIR)):
            if not fname.endswith(".json"):
                continue
            if fname.startswith("distance_matrix"):
                continue
            fpath = os.path.join(_OUTPUT_DIR, fname)
            try:
                data = load_json(fpath)
                if "algorithm" not in data:
                    continue       # skip non-result JSON
                meta = data.get("metadata", {})
                files.append({
                    "filename":        fname,
                    "algorithm":       data.get("algorithm"),
                    "n_clusters":      data.get("n_clusters"),
                    "silhouette_score": data.get("silhouette_score"),
                    "inertia":         data.get("inertia"),
                    "n_documents":     meta.get("n_documents"),
                    "matrix_id":       meta.get("matrix_id"),     # may be None for legacy
                    "has_matrix_info": "matrix_id" in meta,
                })
            except Exception:
                pass
    return jsonify({"results": files})


# ── API: load one result ──────────────────────────────────────────────────────

@app.route("/api/result/<filename>")
def api_result(filename):
    # Safety: prevent directory traversal
    safe = os.path.basename(filename)
    fpath = os.path.join(_OUTPUT_DIR, safe)
    if not os.path.exists(fpath):
        return jsonify({"error": "File not found"}), 404
    data = load_json(fpath)
    return jsonify(data)


# ── API: validate matrix ↔ result match ──────────────────────────────────────

@app.route("/api/validate_matrix")
def api_validate_matrix():
    """Check that a given matrix_id matches a given result filename.

    Query params: ?result=<filename>&matrix_id=<id>
    Returns: {ok: bool, reason: str}
    """
    result_file = request.args.get("result")
    matrix_id   = request.args.get("matrix_id")
    if not result_file or not matrix_id:
        return jsonify({"error": "Provide ?result=<file>&matrix_id=<id>"}), 400

    fpath = os.path.join(_OUTPUT_DIR, os.path.basename(result_file))
    if not os.path.exists(fpath):
        return jsonify({"ok": False, "reason": "Result file not found"}), 404

    data = load_json(fpath)
    stored_id = data.get("metadata", {}).get("matrix_id")
    if stored_id is None:
        return jsonify({"ok": None,
                        "reason": "Legacy result: no matrix_id stored. Cannot verify."})
    if stored_id == matrix_id:
        return jsonify({"ok": True, "reason": "Matrix and result match."})
    return jsonify({
        "ok": False,
        "reason": (
            f"Matrix mismatch. Result was computed with matrix_id={stored_id!r}, "
            f"but provided matrix_id={matrix_id!r}."
        ),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  Project 2 — ClusterLab GUI")
    print("  http://localhost:5000")
    print("=" * 60)
    print()
    app.run(debug=False, port=5000, threaded=True)
