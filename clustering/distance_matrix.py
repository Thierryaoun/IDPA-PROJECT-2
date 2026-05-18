"""
clustering/distance_matrix.py
------------------------------
Pairwise TED computation and distance matrix construction.

Design notes:
  - normalize_country_tree() is called ONCE per tree (not once per pair).
    The n normalized trees are cached in memory before the O(n²) loop starts.
  - Normalized distance ∈ [0, 1] — intersection-based normalization:
        Let S = {second-level field paths present in BOTH A and B}
        d(A, B) = TED(A|S, B|S) / (cost(A|S) + cost(B|S))
    where A|S is tree A restricted to fields in S. A value of 0 means
    identical shared fields; 1 means shared fields are maximally different.
    This avoids the sparse-tree bias of naïve sum normalization: a country
    that omits a field entirely (DELETE cost) no longer appears farther than
    a country that has the field with completely different content (RENAME cost).
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

        stem = fname[:-5]               # strip ".json"  ← stable internal ID

        if filter_set and stem.lower() not in filter_set:
            continue

        try:
            trees[stem] = load_tree_from_json(
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

def _second_level_paths(nt: TreeNode) -> frozenset:
    """
    Return the set of canonical second-level field paths present in a
    normalized tree, e.g. {'demographics.religion', 'languages.official', ...}.

    First-level nodes (demographics, economy, …) are structural containers that
    exist in almost every infobox. Intersecting at the second level captures
    genuine informational overlap — two countries that both report religion or
    both report ethnic_groups — independently of whether each country has a
    history or official-script field.
    """
    paths = set()
    for child in nt.children:
        if child.is_structural():
            for gc in child.children:
                if gc.is_structural():
                    paths.add(f"{child.label}.{gc.label}")
    return frozenset(paths)


def _filter_to_paths(nt: TreeNode, keep: frozenset) -> TreeNode:
    """
    Return a shallow copy of *nt* that retains only those second-level
    structural nodes (and their subtrees) whose path is in *keep*.
    Top-level containers (demographics, economy, …) are kept only when they
    have at least one retained second-level child.

    Only second-level STRUCTURAL grandchildren are considered: a leaf that is a
    direct child of a first-level container is not a second-level path and must
    NOT be retained here — doing so would smuggle fields that are absent from
    the shared intersection into the filtered tree, inflating its cost and
    distorting the TED normalization.
    """
    new_root = TreeNode(nt.label, weight=nt.weight)
    for child in nt.children:
        if not child.is_structural():
            continue
        new_child = TreeNode(child.label, weight=child.weight)
        for gc in child.children:
            path = f"{child.label}.{gc.label}"
            if gc.is_structural() and path in keep:
                new_child.children.append(gc)  # keeps entire subtree
            # NOTE: direct leaf children of first-level nodes are intentionally
            # NOT kept here.  They are not second-level structural paths and
            # therefore not part of the shared intersection.  Keeping them would
            # add country-specific artifacts (sub-region names, territory
            # officer names, etc.) that are not present in the other tree,
            # causing spurious deletion costs and inflated distances.
        if new_child.children:
            new_root.children.append(new_child)
    return new_root


def compute_normalized_ted(
    nt1: TreeNode,
    nt2: TreeNode,
) -> float:
    """
    Compute the intersection-normalized TED distance between two
    *already-normalized* country trees.

    Intersection normalization:
      1. Find the set of second-level field paths present in BOTH trees.
      2. Restrict each tree to those shared fields.
      3. d = TED(filtered_A, filtered_B) / (cost(filtered_A) + cost(filtered_B))

    This avoids the sparse-tree bias in the naïve formula
    d = TED(A,B) / (cost(A) + cost(B)):
    with naïve normalization a country that omits a field (e.g. Germany's
    infobox has no religion entry) is penalised via a DELETE cost, which can
    be cheaper than a full content mismatch rename.  Intersection normalization
    instead ignores unshared fields entirely, so the distance reflects how
    different the *comparable* information actually is.

    Returns a value in [0, 1].
    """
    shared = _second_level_paths(nt1) & _second_level_paths(nt2)
    if not shared:
        return 1.0

    ft1 = _filter_to_paths(nt1, shared)
    ft2 = _filter_to_paths(nt2, shared)

    raw = tree_edit_distance(ft1, ft2)
    denom = weighted_tree_cost(ft1) + weighted_tree_cost(ft2)
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
    field_filter: set | None = None,
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
    field_filter : optional set of canonical field prefixes to include in TED.
                   None means all fields (default). Only applies in semantic mode.
                   Example: {"economy.gdp", "geography.area"}

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
        if field_filter is not None:
            logger.info("Pre-normalizing trees (field filter: %s)…", sorted(field_filter))
        else:
            logger.info("Pre-normalizing trees…")
        normalized: dict[str, TreeNode] = {}
        for name in labels:
            try:
                normalized[name] = normalize_country_tree(trees[name], field_filter=field_filter)
            except Exception as exc:
                logger.warning("Normalization failed for %r: %s — using raw tree", name, exc)
                normalized[name] = trees[name]
    else:
        if field_filter is not None:
            logger.warning("field_filter has no effect in non-semantic mode (use_semantic=False).")
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

    dm = DistanceMatrix(labels=labels, matrix=matrix, field_set=field_filter)
    logger.info("Matrix stats: %s", dm.stats)

    # ── Cache to disk ─────────────────────────────────────────────────────────
    if cache_path:
        dm.save(cache_path)

    return dm
