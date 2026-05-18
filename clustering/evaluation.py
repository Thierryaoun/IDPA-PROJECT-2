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
    For each cluster:
      avg_distance_to_medoid  — mean distance from each member to its medoid
      max_distance_to_medoid  — max distance from any member to the medoid
      avg_pairwise_distance   — mean of all pairwise within-cluster distances
      diameter                — max pairwise distance within the cluster
      closest_pair            — {a, b, distance} of the closest intra-cluster pair
      farthest_pair           — {a, b, distance} of the farthest intra-cluster pair

inter_cluster_distances
    Mean distance between each pair of clusters (medoid-to-medoid).

nearest_neighbors
    For every document, its N nearest neighbours in the full distance matrix
    (regardless of cluster membership), annotated with same_cluster flag.

find_best_k
    Try clustering for each k in a given range, collect silhouette scores,
    and return the ranked list so the caller can pick the optimal k.
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
      avg_distance_to_medoid  — mean distance from each member to the cluster medoid
      max_distance_to_medoid  — max distance from any member to the medoid
      avg_pairwise_distance   — true mean of all pairwise within-cluster distances
      diameter                — max pairwise distance between any two cluster members
      closest_pair            — {a, b, distance} of the two most-similar members
      farthest_pair           — {a, b, distance} of the two most-dissimilar members
      size                    — number of members
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
                "closest_pair":  None,
                "farthest_pair": None,
            }
            continue

        # avg / max distance from each member to the medoid
        dists_to_medoid = [dm.get(i, medoid_idx) for i in indices]
        avg_to_medoid = sum(dists_to_medoid) / size
        max_to_medoid = max(dists_to_medoid)

        # All unique pairwise distances within the cluster
        pairs = [
            (indices[a], indices[b])
            for a in range(len(indices))
            for b in range(a + 1, len(indices))
        ]
        pairwise_dists = [dm.get(ia, ib) for ia, ib in pairs]
        avg_pairwise = sum(pairwise_dists) / len(pairwise_dists)
        diameter     = max(pairwise_dists)

        # Closest and farthest pairs (by member names)
        min_d = min(pairwise_dists)
        max_d = diameter
        min_pair = pairs[pairwise_dists.index(min_d)]
        max_pair = pairs[pairwise_dists.index(max_d)]
        closest_pair  = {
            "a": dm.labels[min_pair[0]],
            "b": dm.labels[min_pair[1]],
            "distance": round(min_d, 6),
        }
        farthest_pair = {
            "a": dm.labels[max_pair[0]],
            "b": dm.labels[max_pair[1]],
            "distance": round(max_d, 6),
        }

        stats[cid] = {
            "size": size,
            "avg_distance_to_medoid": round(avg_to_medoid, 6),
            "max_distance_to_medoid": round(max_to_medoid, 6),
            "avg_pairwise_distance":  round(avg_pairwise, 6),
            "diameter":               round(diameter, 6),
            "closest_pair":           closest_pair,
            "farthest_pair":          farthest_pair,
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
# Nearest neighbours per document
# ─────────────────────────────────────────────────────────────────────────────

def nearest_neighbors(
    dm: DistanceMatrix,
    result: ClusterResult,
    n_neighbors: int = 5,
) -> Dict[str, List[dict]]:
    """
    For every document, return its *n_neighbors* nearest neighbours in the
    full distance matrix (including members of other clusters).

    Returns
    -------
    dict mapping label → list of {label, distance, same_cluster, cluster_id}
    sorted by ascending distance.  The document itself is excluded.
    """
    # Build label → cluster_id lookup
    label_to_cid: Dict[str, int] = {
        a.label: a.cluster_id for a in result.assignments
    }

    nn: Dict[str, List[dict]] = {}
    for i, label_i in enumerate(dm.labels):
        own_cid = label_to_cid.get(label_i)
        candidates = []
        for j, label_j in enumerate(dm.labels):
            if j == i:
                continue
            d = dm.get(i, j)
            candidates.append({
                "label":        label_j,
                "distance":     round(d, 6),
                "same_cluster": label_to_cid.get(label_j) == own_cid,
                "cluster_id":   label_to_cid.get(label_j),
            })
        candidates.sort(key=lambda x: x["distance"])
        nn[label_i] = candidates[:n_neighbors]

    return nn


# ─────────────────────────────────────────────────────────────────────────────
# Best-k selection via silhouette sweep
# ─────────────────────────────────────────────────────────────────────────────

def find_best_k(
    dm: DistanceMatrix,
    algorithm: str = "kmedoids",
    k_range=None,
    linkage: str = "average",
    init_method: str = "heuristic",
    seed: Optional[int] = 42,
) -> dict:
    """
    Try each k in *k_range* and return silhouette scores and a recommendation.

    Parameters
    ----------
    dm         : pre-computed distance matrix
    algorithm  : "kmedoids" | "hierarchical"
    k_range    : iterable of k values to try (default: 2 to min(15, n-1))
    linkage    : linkage for HAC (ignored for k-medoids)
    init_method: initialisation for k-medoids
    seed       : random seed for k-medoids

    Returns
    -------
    {
      "scores":      [{k, silhouette, inertia}, ...] sorted by k,
      "best_k":      int  — k with highest silhouette,
      "best_sil":    float,
      "algorithm":   str,
    }
    """
    from .kmedoids import kmedoids
    from .hierarchical import hierarchical_clustering

    if k_range is None:
        k_range = range(2, min(16, dm.n))

    scores = []
    for k in k_range:
        if k < 1 or k > dm.n:
            continue
        try:
            if algorithm == "kmedoids":
                # Use 3 restarts in the sweep to keep it fast while still escaping bad inits
                result = kmedoids(dm, n_clusters=k, init_method=init_method, seed=seed, n_init=3)
            else:
                result = hierarchical_clustering(dm, n_clusters=k, linkage=linkage)

            sil = silhouette_score(dm, result)
            scores.append({
                "k":          k,
                "silhouette": round(sil, 6) if math.isfinite(sil) else None,
                "inertia":    round(result.inertia, 6) if result.inertia is not None else None,
            })
            logger.debug("find_best_k: k=%d  sil=%.4f", k, sil if math.isfinite(sil) else float("nan"))
        except Exception as exc:
            logger.warning("find_best_k: k=%d failed: %s", k, exc)
            scores.append({"k": k, "silhouette": None, "inertia": None})

    valid = [s for s in scores if s["silhouette"] is not None]
    if valid:
        best = max(valid, key=lambda s: s["silhouette"])
        best_k   = best["k"]
        best_sil = best["silhouette"]
    else:
        best_k   = list(k_range)[0] if hasattr(k_range, "__iter__") else 2
        best_sil = None

    return {
        "scores":    scores,
        "best_k":    best_k,
        "best_sil":  best_sil,
        "algorithm": algorithm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined summary — mutates ClusterResult in-place
# ─────────────────────────────────────────────────────────────────────────────

def attach_evaluation(
    dm: DistanceMatrix,
    result: ClusterResult,
    include_neighbors: bool = True,
    n_neighbors: int = 5,
) -> ClusterResult:
    """
    Compute all evaluation metrics and attach them to *result* in-place.

    Sets:
      result.silhouette_score
      result.cluster_stats   (intra-cluster statistics per cluster,
                               including closest_pair and farthest_pair)
      result.metadata["inter_cluster_distances"]
      result.metadata["nearest_neighbors"]   (if include_neighbors=True)

    Returns the same result object (mutated) for fluent chaining.
    """
    logger.info("Computing evaluation metrics for %s result…", result.algorithm)

    result.silhouette_score = silhouette_score(dm, result)
    result.cluster_stats    = intra_cluster_distances(dm, result)
    result.metadata["inter_cluster_distances"] = inter_cluster_distances(dm, result)

    if include_neighbors:
        result.metadata["nearest_neighbors"] = nearest_neighbors(
            dm, result, n_neighbors=n_neighbors
        )

    sil_val = result.silhouette_score
    logger.info(
        "Evaluation complete.  silhouette=%.4f  inertia=%.4f",
        sil_val if (sil_val == sil_val) else 0.0,
        result.inertia or 0,
    )
    return result
