"""
clustering/distance_matrix.py
------------------------------
Pairwise TED computation and distance matrix construction.

Design notes:
  - normalize_country_tree() is called ONCE per tree (not once per pair).
    The n normalized trees are cached in memory before the O(n²) loop starts.
  - Normalized distance ∈ [0, 1]:
        d(A, B) = TED(A, B) / (weighted_tree_cost(A) + weighted_tree_cost(B))
    A value of 0 means identical trees; 1 means no shared structure/content.
  - The completed matrix is saved as JSON so the expensive computation is
    skipped on subsequent runs (controlled by the `cache_path` argument).
  - Progress is reported via ProgressReporter so long runs are observable.
"""

import os
import sys

# Make sure the project root is on the path so ted_algorithm is importable
# regardless of how this package is invoked.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ted_algorithm as ted
from ted_algorithm import (
    TreeNode,
    load_tree_from_json,
    normalize_country_tree,
    tree_edit_distance,
    tree_edit_distance_legacy,
    weighted_tree_cost,
)

from .models import DistanceMatrix
from .utils import get_logger, ProgressReporter

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tree loading
# ─────────────────────────────────────────────────────────────────────────────

def load_country_trees(
    tree_dir: str,
    country_filter: list[str] | None = None,
) -> dict[str, TreeNode]:
    """
    Load preprocessed country trees from *tree_dir*.

    Parameters
    ----------
    tree_dir       : path to preprocessed_trees/ directory
    country_filter : if given, only load countries whose names appear in
                     this list (case-insensitive match on the filename stem)

    Returns
    -------
    dict mapping country name → TreeNode  (sorted alphabetically)
    """
    if not os.path.isdir(tree_dir):
        raise FileNotFoundError(f"Tree directory not found: {tree_dir!r}")

    filter_set = (
        {n.lower().replace(" ", "_") for n in country_filter}
        if country_filter else None
    )

    trees: dict[str, TreeNode] = {}
    failed: list[str] = []

    for fname in sorted(os.listdir(tree_dir)):
        if not fname.endswith(".json") or fname.startswith("_"):
            continue

        stem = fname[:-5]               # strip ".json"
        display_name = stem.replace("_", " ")

        if filter_set and stem.lower() not in filter_set:
            continue

        try:
            trees[display_name] = load_tree_from_json(
                os.path.join(tree_dir, fname)
            )
        except Exception as exc:
            logger.warning("Could not load %s: %s", fname, exc)
            failed.append(fname)

    if failed:
        logger.warning("%d file(s) failed to load: %s", len(failed), failed)

    logger.info("Loaded %d country trees from %r", len(trees), tree_dir)
    return trees


# ─────────────────────────────────────────────────────────────────────────────
# Single-pair distance
# ─────────────────────────────────────────────────────────────────────────────

def compute_normalized_ted(
    nt1: TreeNode,
    nt2: TreeNode,
) -> float:
    """
    Compute the normalized TED distance between two *already-normalized* trees.

    The result is guaranteed in [0, 1]:
        d = TED(nt1, nt2) / (cost(nt1) + cost(nt2))

    Pass pre-normalized trees (output of normalize_country_tree) to avoid
    redundant normalization inside the O(n²) loop.
    """
    raw = tree_edit_distance(nt1, nt2)
    denom = weighted_tree_cost(nt1) + weighted_tree_cost(nt2)
    if denom <= 0:
        return 0.0
    return min(1.0, raw / denom)


def compute_normalized_ted_legacy(
    t1: TreeNode,
    t2: TreeNode,
) -> float:
    """
    Compute normalized TED with the unit (unweighted) cost model.

    Used for raw/baseline distance.  Trees do NOT need to be pre-normalized.
    """
    raw = tree_edit_distance_legacy(t1, t2)
    denom = t1.size() + t2.size()
    if denom <= 0:
        return 0.0
    return min(1.0, raw / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Full distance matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_distance_matrix(
    trees: dict[str, TreeNode],
    use_semantic: bool = True,
    cache_path: str | None = None,
) -> DistanceMatrix:
    """
    Compute the full pairwise distance matrix for a collection of trees.

    Parameters
    ----------
    trees        : dict returned by load_country_trees()
    use_semantic : if True, use weighted semantic TED (recommended);
                   if False, use raw unit-cost TED (faster, less accurate)
    cache_path   : if given, save the matrix here after computation and
                   load from here on subsequent calls (skip recomputation)

    Returns
    -------
    DistanceMatrix  (n×n symmetric, diagonal=0, values in [0, 1])
    """
    # ── Try cache first ───────────────────────────────────────────────────────
    if cache_path and os.path.exists(cache_path):
        logger.info("Loading cached distance matrix from %r", cache_path)
        dm = DistanceMatrix.load(cache_path)
        # Validate that cached labels match current tree set
        if set(dm.labels) == set(trees.keys()):
            return dm
        logger.warning(
            "Cached matrix labels differ from loaded trees — recomputing."
        )

    labels = sorted(trees.keys())
    n = len(labels)
    logger.info(
        "Computing %s distance matrix for %d documents (%d pairs)…",
        "semantic" if use_semantic else "raw",
        n,
        n * (n - 1) // 2,
    )

    # ── Pre-normalize all trees once (semantic mode) ──────────────────────────
    if use_semantic:
        logger.info("Pre-normalizing trees…")
        normalized: dict[str, TreeNode] = {}
        for name in labels:
            try:
                normalized[name] = normalize_country_tree(trees[name])
            except Exception as exc:
                logger.warning("Normalization failed for %r: %s — using raw tree", name, exc)
                normalized[name] = trees[name]
    else:
        normalized = {name: trees[name] for name in labels}

    # ── Pairwise computation ──────────────────────────────────────────────────
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    n_pairs = n * (n - 1) // 2
    reporter = ProgressReporter(total=n_pairs, label="Pairwise TED")

    for i in range(n):
        for j in range(i + 1, n):
            try:
                if use_semantic:
                    d = compute_normalized_ted(normalized[labels[i]], normalized[labels[j]])
                else:
                    d = compute_normalized_ted_legacy(normalized[labels[i]], normalized[labels[j]])
            except Exception as exc:
                logger.warning(
                    "TED failed for (%s, %s): %s — using 1.0",
                    labels[i], labels[j], exc,
                )
                d = 1.0

            matrix[i][j] = d
            matrix[j][i] = d
            reporter.update()

    reporter.done()

    dm = DistanceMatrix(labels=labels, matrix=matrix)
    logger.info("Matrix stats: %s", dm.stats)

    # ── Cache to disk ─────────────────────────────────────────────────────────
    if cache_path:
        dm.save(cache_path)

    return dm
