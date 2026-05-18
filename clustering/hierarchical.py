"""
clustering/hierarchical.py
--------------------------
Hierarchical Agglomerative Clustering (HAC) over a DistanceMatrix.

Algorithm summary
-----------------
1. Start with n singleton clusters (one per document).
2. At each step, find the two clusters whose *linkage distance* is smallest.
3. Merge them into a new cluster.
4. Repeat until exactly `n_clusters` clusters remain.

Linkage methods
---------------
single   — minimum distance between any cross-cluster pair ("nearest neighbour")
complete — maximum distance between any cross-cluster pair ("furthest neighbour")
average  — mean of all pairwise cross-cluster distances (UPGMA)

Ward's linkage is intentionally NOT offered because it requires a Euclidean
embedding, which TED distances do not provide.

Medoid selection
----------------
For each final cluster, the medoid is the member that minimises the sum of
distances to every other member in the same cluster.  This is consistent with
the K-Medoids output so both algorithms can be compared fairly.

Complexity
----------
O(n²  × merges)  where merges = n − k.
For n = 193, k = 5, this is ~200 k distance look-ups — fast in practice.
"""

import math
from typing import Dict, List, Optional

from .models import ClusterAssignment, ClusterResult, DistanceMatrix
from .utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────────

def hierarchical_clustering(
    dm: DistanceMatrix,
    n_clusters: int = 5,
    linkage: str = "average",
) -> ClusterResult:
    """
    Run HAC on *dm* until *n_clusters* clusters remain.

    Parameters
    ----------
    dm         : pre-computed distance matrix
    n_clusters : desired number of output clusters (1 ≤ n_clusters ≤ dm.n)
    linkage    : "single" | "complete" | "average"

    Returns
    -------
    ClusterResult with medoids, assignments, and merge history in metadata.
    """
    if linkage not in ("single", "complete", "average"):
        raise ValueError(f"Unknown linkage {linkage!r}. Choose single/complete/average.")
    if dm.n == 0:
        raise ValueError("Distance matrix is empty (n=0).")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}.")
    if n_clusters > dm.n:
        raise ValueError(
            f"n_clusters={n_clusters} exceeds number of documents n={dm.n}."
        )
    logger.info(
        "HAC: n=%d, k=%d, linkage=%s", dm.n, n_clusters, linkage
    )

    # ── Initialise: each document is its own cluster ─────────────────────────
    # clusters: {cluster_id: [doc_indices]}
    clusters: Dict[int, List[int]] = {i: [i] for i in range(dm.n)}
    merge_history: List[dict] = []
    next_id = dm.n   # new cluster IDs start after the original singleton IDs

    # ── Greedy merging ───────────────────────────────────────────────────────
    while len(clusters) > n_clusters:
        cluster_ids = list(clusters.keys())
        best_dist = math.inf
        best_pair = (cluster_ids[0], cluster_ids[1])

        for a_idx in range(len(cluster_ids)):
            for b_idx in range(a_idx + 1, len(cluster_ids)):
                a_id = cluster_ids[a_idx]
                b_id = cluster_ids[b_idx]
                d = _linkage_distance(
                    dm,
                    clusters[a_id],
                    clusters[b_id],
                    linkage,
                )
                if d < best_dist:
                    best_dist = d
                    best_pair = (a_id, b_id)

        a_id, b_id = best_pair
        merged_members = clusters[a_id] + clusters[b_id]
        clusters[next_id] = merged_members
        del clusters[a_id]
        del clusters[b_id]

        merge_history.append({
            "step":     dm.n - len(clusters),
            "merged":   [a_id, b_id],
            "new_id":   next_id,
            "distance": round(best_dist, 6),
            "size":     len(merged_members),
        })
        logger.debug(
            "  Merged clusters %d+%d → %d  (dist=%.4f, size=%d)",
            a_id, b_id, next_id, best_dist, len(merged_members),
        )
        next_id += 1

    # ── Assign final cluster IDs 0…k-1 ───────────────────────────────────────
    final_clusters: Dict[int, List[int]] = {
        new_cid: members
        for new_cid, (_, members) in enumerate(sorted(clusters.items()))
    }

    # ── Find medoid for each cluster ──────────────────────────────────────────
    medoid_indices: Dict[int, int] = {
        cid: _find_medoid(dm, members)
        for cid, members in final_clusters.items()
    }

    # ── Build ClusterAssignment list ──────────────────────────────────────────
    assignments: List[ClusterAssignment] = []
    for cid, members in final_clusters.items():
        medoid_idx = medoid_indices[cid]
        medoid_name = dm.labels[medoid_idx]
        for idx in members:
            assignments.append(ClusterAssignment(
                label=dm.labels[idx],
                cluster_id=cid,
                medoid=medoid_name,
                distance_to_medoid=dm.get(idx, medoid_idx),
            ))

    # ── Assemble result ───────────────────────────────────────────────────────
    clusters_by_name: Dict[int, List[str]] = {
        cid: [dm.labels[i] for i in members]
        for cid, members in final_clusters.items()
    }
    medoids_by_name: Dict[int, str] = {
        cid: dm.labels[idx] for cid, idx in medoid_indices.items()
    }
    inertia = sum(a.distance_to_medoid for a in assignments)

    result = ClusterResult(
        algorithm="hierarchical",
        n_clusters=n_clusters,
        assignments=assignments,
        clusters=clusters_by_name,
        medoids=medoids_by_name,
        linkage=linkage,
        inertia=round(inertia, 6),
        metadata={
            "merge_history":  merge_history,
            # Matrix provenance — used by frontend to refuse mismatched display
            "matrix_id":      dm.matrix_id,
            "matrix_labels":  dm.labels,
        },
    )
    logger.info("HAC complete.  Inertia=%.4f", inertia)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _linkage_distance(
    dm: DistanceMatrix,
    members_a: List[int],
    members_b: List[int],
    linkage: str,
) -> float:
    """
    Compute the linkage distance between two sets of document indices.

    single   → min over all (i∈A, j∈B) of d(i,j)
    complete → max over all (i∈A, j∈B) of d(i,j)
    average  → mean over all (i∈A, j∈B) of d(i,j)
    """
    cross_distances = [
        dm.get(i, j)
        for i in members_a
        for j in members_b
    ]
    if not cross_distances:
        return 0.0

    if linkage == "single":
        return min(cross_distances)
    if linkage == "complete":
        return max(cross_distances)
    # average (UPGMA)
    return sum(cross_distances) / len(cross_distances)


def _find_medoid(dm: DistanceMatrix, indices: List[int]) -> int:
    """
    Return the index (within *indices*) of the medoid.

    The medoid is the document that minimises the sum of distances to all
    other members of the cluster.
    """
    if len(indices) == 1:
        return indices[0]
    return min(
        indices,
        key=lambda i: sum(dm.get(i, j) for j in indices if j != i),
    )
