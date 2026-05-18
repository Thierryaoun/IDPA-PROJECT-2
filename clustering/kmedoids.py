"""
clustering/kmedoids.py
----------------------
K-Medoids clustering using the PAM (Partitioning Around Medoids) algorithm.

Why K-Medoids (not K-Means)?
-----------------------------
K-Means requires a notion of "centroid" — the mean of a set of points.
TED distances do not live in a Euclidean space, so there is no meaningful
centroid.  K-Medoids uses an *actual document* from the dataset as the
cluster representative (the medoid), making it directly applicable to
any distance matrix.

PAM algorithm
-------------
Phase 1 — INITIALISE
  Select k initial medoids.  Two strategies are offered:
    "random"     — pick k random indices (reproducible with seed)
    "heuristic"  — greedy furthest-first: pick the global medoid as the
                   first medoid, then repeatedly pick the point that
                   maximises the minimum distance to already-chosen medoids.
                   More stable than random, less sensitive to seed choice.

Phase 2 — ASSIGN
  Assign every non-medoid document to its closest medoid.

Phase 3 — UPDATE (SWAP)
  For every medoid m and every non-medoid candidate c:
    Tentatively swap m ↔ c.
    Compute the new total cost (sum of distances to assigned medoids).
    Keep the swap if it strictly reduces the total cost.
  Repeat Assign + Update until no swap reduces the cost (convergence).

Complexity
----------
Each SWAP iteration: O(k × (n−k)) candidate swaps, each O(n) to evaluate.
Total: O(k(n−k)·n · max_iter).
For n=193, k=5: ≈ 5 × 188 × 193 × iters ≈ 181 k distance look-ups per iter.
Typically converges in < 20 iterations.
"""

import random
from typing import Dict, List, Optional, Tuple

from .models import ClusterAssignment, ClusterResult, DistanceMatrix
from .utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────────

def kmedoids(
    dm: DistanceMatrix,
    n_clusters: int = 5,
    max_iter: int = 100,
    init_method: str = "heuristic",
    seed: Optional[int] = 42,
) -> ClusterResult:
    """
    Run K-Medoids (PAM) on *dm*.

    Parameters
    ----------
    dm          : pre-computed distance matrix
    n_clusters  : number of clusters k  (1 ≤ k ≤ dm.n)
    max_iter    : maximum SWAP iterations before forced stop
    init_method : "heuristic" (recommended) | "random"
    seed        : random seed (used only by "random" init)

    Returns
    -------
    ClusterResult with medoids, assignments, inertia, and convergence info.
    """
    n_clusters = max(1, min(n_clusters, dm.n))
    logger.info(
        "K-Medoids: n=%d, k=%d, init=%s, max_iter=%d",
        dm.n, n_clusters, init_method, max_iter,
    )

    # ── Phase 1: Initialise medoids ──────────────────────────────────────────
    medoid_indices: List[int] = _initialize_medoids(dm, n_clusters, init_method, seed)
    logger.debug("Initial medoids: %s", [dm.labels[m] for m in medoid_indices])

    prev_inertia = float("inf")
    iterations_run = 0
    converged = False

    for iteration in range(max_iter):
        iterations_run = iteration + 1

        # ── Phase 2: Assign each document to its nearest medoid ──────────────
        assignments_map: List[int] = _assign_clusters(dm, medoid_indices)
        inertia = _compute_inertia(dm, medoid_indices, assignments_map)
        logger.debug("  iter %d  inertia=%.6f", iteration, inertia)

        # ── Phase 3: SWAP — try every (medoid, non-medoid) pair ──────────────
        improved = False
        for m_pos, m_idx in enumerate(medoid_indices):
            for c_idx in range(dm.n):
                if c_idx in medoid_indices:
                    continue

                # Tentative new medoid set with m_idx replaced by c_idx
                candidate_medoids = medoid_indices[:m_pos] + [c_idx] + medoid_indices[m_pos + 1:]
                candidate_assignments = _assign_clusters(dm, candidate_medoids)
                candidate_inertia = _compute_inertia(dm, candidate_medoids, candidate_assignments)

                if candidate_inertia < inertia - 1e-9:
                    medoid_indices = candidate_medoids
                    assignments_map = candidate_assignments
                    inertia = candidate_inertia
                    improved = True
                    logger.debug(
                        "    Swap medoid[%d]=%s → %s  (inertia=%.6f)",
                        m_pos, dm.labels[m_idx], dm.labels[c_idx], inertia,
                    )
                    # Restart the swap loop with the improved medoid set
                    break
            if improved:
                break

        if not improved:
            converged = True
            logger.info(
                "K-Medoids converged in %d iteration(s).  Inertia=%.6f",
                iterations_run, inertia,
            )
            break

    if not converged:
        logger.warning(
            "K-Medoids reached max_iter=%d without full convergence.  "
            "Inertia=%.6f",
            max_iter, inertia,
        )

    # ── Build output structures ───────────────────────────────────────────────
    final_assignments_map = _assign_clusters(dm, medoid_indices)

    # Map medoid index → cluster id (0-based, sorted by medoid label for determinism)
    sorted_medoids = sorted(enumerate(medoid_indices), key=lambda x: dm.labels[x[1]])
    medoid_to_cluster: Dict[int, int] = {
        m_idx: new_cid
        for new_cid, (_, m_idx) in enumerate(sorted_medoids)
    }

    clusters: Dict[int, List[str]] = {cid: [] for cid in range(n_clusters)}
    medoids_by_name: Dict[int, str] = {
        cid: dm.labels[m_idx]
        for m_idx, cid in medoid_to_cluster.items()
    }
    assignments: List[ClusterAssignment] = []

    for doc_idx, nearest_medoid_idx in enumerate(final_assignments_map):
        cid = medoid_to_cluster[nearest_medoid_idx]
        clusters[cid].append(dm.labels[doc_idx])
        assignments.append(ClusterAssignment(
            label=dm.labels[doc_idx],
            cluster_id=cid,
            medoid=dm.labels[nearest_medoid_idx],
            distance_to_medoid=dm.get(doc_idx, nearest_medoid_idx),
        ))

    final_inertia = sum(a.distance_to_medoid for a in assignments)

    result = ClusterResult(
        algorithm="kmedoids",
        n_clusters=n_clusters,
        assignments=assignments,
        clusters=clusters,
        medoids=medoids_by_name,
        inertia=round(final_inertia, 6),
        metadata={
            "iterations": iterations_run,
            "converged": converged,
            "init_method": init_method,
            "seed": seed,
        },
    )
    logger.info(
        "K-Medoids complete.  k=%d, inertia=%.4f, iters=%d",
        n_clusters, final_inertia, iterations_run,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _initialize_medoids(
    dm: DistanceMatrix,
    k: int,
    method: str,
    seed: Optional[int],
) -> List[int]:
    """
    Choose k initial medoid indices.

    heuristic (greedy furthest-first):
      1. Pick the global medoid (lowest sum of distances to all others) as m0.
      2. For each subsequent medoid, pick the non-medoid point that maximises
         its minimum distance to any already-chosen medoid.
    This spreads the initial medoids across the space and avoids the poor
    local optima common with random initialisation.

    random:
      Simply sample k distinct indices.
    """
    if method == "random":
        rng = random.Random(seed)
        return rng.sample(range(dm.n), k)

    # heuristic: greedy furthest-first
    # Step 1 — global medoid
    first = min(
        range(dm.n),
        key=lambda i: sum(dm.get(i, j) for j in range(dm.n)),
    )
    chosen = [first]

    # Step 2 — each subsequent medoid maximises min-distance to chosen set
    while len(chosen) < k:
        best_idx = -1
        best_min_dist = -1.0
        for c in range(dm.n):
            if c in chosen:
                continue
            min_dist_to_chosen = min(dm.get(c, m) for m in chosen)
            if min_dist_to_chosen > best_min_dist:
                best_min_dist = min_dist_to_chosen
                best_idx = c
        chosen.append(best_idx)

    return chosen


def _assign_clusters(
    dm: DistanceMatrix,
    medoid_indices: List[int],
) -> List[int]:
    """
    Assign every document to its nearest medoid.

    Returns a list of length dm.n where element i is the medoid index
    (from medoid_indices) that document i is assigned to.
    """
    assignments: List[int] = []
    for i in range(dm.n):
        nearest = min(medoid_indices, key=lambda m: dm.get(i, m))
        assignments.append(nearest)
    return assignments


def _compute_inertia(
    dm: DistanceMatrix,
    medoid_indices: List[int],
    assignments: List[int],
) -> float:
    """
    Total cost = sum of distances from every document to its assigned medoid.
    """
    return sum(dm.get(i, assignments[i]) for i in range(dm.n))
