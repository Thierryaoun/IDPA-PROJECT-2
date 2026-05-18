"""
clustering/evaluation.py
------------------------
Quality metrics for clustering results.

Metrics
-------
silhouette_score
    For each document i:
      a(i) = mean distance to other members of its cluster
      b(i) = min over all other clusters c of (mean distance to members of c)
      s(i) = (b(i) - a(i)) / max(a(i), b(i))
    The overall silhouette score is the mean of s(i) over all documents.
    Range: [-1, +1].  Higher is better.  < 0 means a document is likely
    in the wrong cluster.  Requires at least 2 clusters and 2 members per
    cluster to be meaningful.

intra_cluster_distances
    For each cluster: avg, max, and diameter of intra-cluster distances.
    Diameter = max pairwise distance within the cluster.

inter_cluster_distances
    Mean distance between each pair of clusters (using medoids).

cluster_summary
    Combines intra + inter metrics and attaches them to a ClusterResult
    by populating result.cluster_stats and result.silhouette_score.
"""

import math
from typing import Dict, List, Optional

from .models import ClusterResult, DistanceMatrix
from .utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Silhouette score
# ─────────────────────────────────────────────────────────────────────────────

def silhouette_score(dm: DistanceMatrix, result: ClusterResult) -> float:
    """
    Compute the mean silhouette coefficient for a clustering result.

    Returns NaN if the score is undefined (single cluster, or any cluster
    with only one member and no well-defined b(i)).
    """
    if result.n_clusters < 2:
        logger.warning("Silhouette score is undefined for < 2 clusters.")
        return float("nan")

    # Build index → cluster_id lookup
    label_to_cid: Dict[str, int] = {
        a.label: a.cluster_id for a in result.assignments
    }
    # Build cluster_id → list of doc indices
    cid_to_indices: Dict[int, List[int]] = {}
    for a in result.assignments:
        idx = dm.index_of(a.label)
        cid_to_indices.setdefault(a.cluster_id, []).append(idx)

    scores: List[float] = []

    for a in result.assignments:
        i = dm.index_of(a.label)
        own_cid = a.cluster_id
        own_members = cid_to_indices[own_cid]

        # a(i): mean distance to other members of same cluster
        others_in_cluster = [j for j in own_members if j != i]
        if not others_in_cluster:
            # Singleton cluster — silhouette is 0 by convention
            scores.append(0.0)
            continue
        a_i = sum(dm.get(i, j) for j in others_in_cluster) / len(others_in_cluster)

        # b(i): min mean distance to any *other* cluster
        b_i = math.inf
        for other_cid, other_members in cid_to_indices.items():
            if other_cid == own_cid:
                continue
            mean_dist = sum(dm.get(i, j) for j in other_members) / len(other_members)
            b_i = min(b_i, mean_dist)

        if b_i == math.inf:
            # Only one cluster exists somehow
            scores.append(0.0)
            continue

        denom = max(a_i, b_i)
        s_i = (b_i - a_i) / denom if denom > 0 else 0.0
        scores.append(s_i)

    return round(sum(scores) / len(scores), 6) if scores else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Intra-cluster statistics
# ─────────────────────────────────────────────────────────────────────────────

def intra_cluster_distances(
    dm: DistanceMatrix,
    result: ClusterResult,
) -> Dict[int, dict]:
    """
    For each cluster compute:
      avg_distance_to_medoid — mean distance from each member to the cluster medoid
      max_distance_to_medoid — max distance from any member to the medoid
      avg_pairwise_distance  — true mean of all pairwise within-cluster distances
      diameter               — max pairwise distance between any two cluster members
      size                   — number of members
    """
    stats: Dict[int, dict] = {}

    for cid, members in result.clusters.items():
        indices = [dm.index_of(name) for name in members]
        medoid_idx = dm.index_of(result.medoids[cid])
        size = len(indices)

        if size == 1:
            stats[cid] = {
                "size": 1,
                "avg_distance_to_medoid": 0.0,
                "max_distance_to_medoid": 0.0,
                "avg_pairwise_distance":  0.0,
                "diameter": 0.0,
            }
            continue

        # avg / max distance from each member to the medoid
        dists_to_medoid = [dm.get(i, medoid_idx) for i in indices]
        avg_to_medoid = sum(dists_to_medoid) / size
        max_to_medoid = max(dists_to_medoid)

        # All unique pairwise distances within the cluster
        pairwise = [
            dm.get(indices[a], indices[b])
            for a in range(len(indices))
            for b in range(a + 1, len(indices))
        ]
        avg_pairwise = sum(pairwise) / len(pairwise) if pairwise else 0.0
        diameter     = max(pairwise) if pairwise else 0.0

        stats[cid] = {
            "size": size,
            "avg_distance_to_medoid": round(avg_to_medoid, 6),
            "max_distance_to_medoid": round(max_to_medoid, 6),
            "avg_pairwise_distance":  round(avg_pairwise, 6),
            "diameter":               round(diameter, 6),
        }

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Inter-cluster statistics
# ─────────────────────────────────────────────────────────────────────────────

def inter_cluster_distances(
    dm: DistanceMatrix,
    result: ClusterResult,
) -> Dict[str, float]:
    """
    Compute the distance between each pair of clusters (medoid-to-medoid).

    Returns a dict keyed by "cid_a,cid_b" strings.
    """
    inter: Dict[str, float] = {}
    cluster_ids = sorted(result.clusters.keys())
    for a in range(len(cluster_ids)):
        for b in range(a + 1, len(cluster_ids)):
            ca, cb = cluster_ids[a], cluster_ids[b]
            medoid_a = dm.index_of(result.medoids[ca])
            medoid_b = dm.index_of(result.medoids[cb])
            key = f"{ca},{cb}"
            inter[key] = round(dm.get(medoid_a, medoid_b), 6)
    return inter


# ─────────────────────────────────────────────────────────────────────────────
# Combined summary — mutates ClusterResult in-place
# ─────────────────────────────────────────────────────────────────────────────

def attach_evaluation(dm: DistanceMatrix, result: ClusterResult) -> ClusterResult:
    """
    Compute all evaluation metrics and attach them to *result* in-place.

    Sets:
      result.silhouette_score
      result.cluster_stats   (intra-cluster statistics per cluster)
      result.metadata["inter_cluster_distances"]

    Returns the same result object (mutated) for fluent chaining.
    """
    logger.info("Computing evaluation metrics for %s result…", result.algorithm)

    result.silhouette_score = silhouette_score(dm, result)
    result.cluster_stats    = intra_cluster_distances(dm, result)
    result.metadata["inter_cluster_distances"] = inter_cluster_distances(dm, result)

    logger.info(
        "Evaluation complete.  silhouette=%.4f  inertia=%.4f",
        result.silhouette_score if result.silhouette_score == result.silhouette_score else 0,
        result.inertia or 0,
    )
    return result
