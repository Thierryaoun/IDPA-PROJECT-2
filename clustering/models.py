"""
clustering/models.py
---------------------
Data classes used throughout the clustering package.

Design notes:
  - DistanceMatrix owns the n×n float grid and its label index.
    It is the single shared input to both clustering algorithms.
  - ClusterResult is the uniform output type for both HAC and K-Medoids,
    making it easy for a GUI layer to consume either result identically.
  - ClusterConfig is a plain configuration bag — pass it to the pipeline
    instead of scattering keyword arguments everywhere.
  - All three classes are JSON-serialisable via .to_dict() / .save().
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .utils import get_logger, save_json, load_json, matrix_stats

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Display-name helper  (stem → readable label for UI only)
# ─────────────────────────────────────────────────────────────────────────────

def display_name(stem: str) -> str:
    """Convert a filename stem to a readable UI label.

    Examples
    --------
    "Lebanon"                         -> "Lebanon"
    "South_Africa"                    -> "South Africa"
    "Bahamas__The"                    -> "Bahamas, The"
    "Iran__Islamic_Republic_of_"      -> "Iran, Islamic Republic of"
    "Gambia__Republic_of_The_"        -> "Gambia, Republic of The"
    """
    # Replace double underscore first (it becomes ", "), then single underscore
    result = stem.replace("__", ", ").replace("_", " ").strip()
    # Strip trailing comma-space if stem ended with underscore groups
    result = result.rstrip(", ").strip()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DistanceMatrix
# ─────────────────────────────────────────────────────────────────────────────

class DistanceMatrix:
    """
    Square symmetric distance matrix over a labelled set of documents.

    Attributes
    ----------
    labels : list[str]
        Ordered list of document names (one per row/column).
    matrix : list[list[float]]
        n×n symmetric matrix; matrix[i][j] == matrix[j][i], diagonal == 0.
    stats  : dict
        Precomputed descriptive statistics (min, max, mean, std).
    """

    def __init__(
        self,
        labels: List[str],
        matrix: List[List[float]],
        skip_validation: bool = False,
        field_set=None,
    ) -> None:
        if len(labels) != len(matrix):
            raise ValueError(
                f"labels length {len(labels)} != matrix rows {len(matrix)}"
            )
        self.labels = labels
        self.matrix = matrix
        self.n = len(labels)
        # Build a fast name→index lookup
        self._index: Dict[str, int] = {lbl: i for i, lbl in enumerate(labels)}
        self.stats = matrix_stats(matrix)
        # Optional set of field prefixes used during TED computation
        self.field_set = frozenset(field_set) if field_set else None
        if not skip_validation:
            self._validate()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """Check that the matrix is square, symmetric, has zero diagonal,
        finite values, and all values in [0, 1].  Raises ValueError on failure.
        """
        n = self.n
        TOL = 1e-8
        for i in range(n):
            row = self.matrix[i]
            if len(row) != n:
                raise ValueError(
                    f"Matrix row {i} has length {len(row)}, expected {n}"
                )
            if abs(row[i]) > TOL:
                raise ValueError(
                    f"Diagonal matrix[{i}][{i}] = {row[i]:.6g} is not zero"
                )
            for j in range(i + 1, n):
                v = row[j]
                if not math.isfinite(v):
                    raise ValueError(
                        f"matrix[{i}][{j}] is not finite: {v}"
                    )
                if v < -TOL or v > 1.0 + TOL:
                    raise ValueError(
                        f"matrix[{i}][{j}] = {v:.6g} is outside [0, 1]"
                    )
                if abs(v - self.matrix[j][i]) > TOL:
                    raise ValueError(
                        f"matrix not symmetric: [{i}][{j}]={v:.6g} "
                        f"but [{j}][{i}]={self.matrix[j][i]:.6g}"
                    )

    @property
    def matrix_id(self) -> str:
        """Stable short identifier based on sorted labels and selected fields.

        Used to match a result to its source matrix without storing
        the full matrix inside every result file.
        """
        import hashlib
        key = ",".join(sorted(self.labels))
        if self.field_set:
            key += "|" + ",".join(sorted(self.field_set))
        return hashlib.md5(key.encode()).hexdigest()[:12]

    # ── Access ────────────────────────────────────────────────────────────────

    def get(self, i: int, j: int) -> float:
        """Return the distance between document at index *i* and index *j*."""
        return self.matrix[i][j]

    def distance(self, name_a: str, name_b: str) -> float:
        """Return the distance between two documents by name."""
        return self.matrix[self._index[name_a]][self._index[name_b]]

    def index_of(self, name: str) -> int:
        """Return the numeric index for a document name."""
        return self._index[name]

    def row(self, i: int) -> List[float]:
        """Return all distances from document *i* to every other document."""
        return self.matrix[i]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = {
            "labels":    self.labels,
            "matrix":    self.matrix,
            "stats":     self.stats,
            "matrix_id": self.matrix_id,
        }
        if self.field_set:
            d["selected_fields"] = sorted(self.field_set)
        return d

    def save(self, path: str) -> None:
        """Persist to a JSON file so the expensive computation is not repeated."""
        save_json(self.to_dict(), path)
        logger.info("Distance matrix saved -> %s  (%dx%d)", path, self.n, self.n)

    @classmethod
    def load(cls, path: str) -> "DistanceMatrix":
        """Restore a previously saved distance matrix (skips full re-validation)."""
        data = load_json(path)
        field_set = data.get("selected_fields")  # None for legacy matrices
        dm = cls(
            labels=data["labels"],
            matrix=data["matrix"],
            skip_validation=True,
            field_set=field_set,
        )
        logger.info("Distance matrix loaded <- %s  (%dx%d)", path, dm.n, dm.n)
        return dm

    # ── Subset ────────────────────────────────────────────────────────────────

    def subset(self, names: List[str]) -> "DistanceMatrix":
        """Return a new DistanceMatrix restricted to the given country names."""
        indices = [self._index[n] for n in names]
        sub_matrix = [
            [self.matrix[i][j] for j in indices]
            for i in indices
        ]
        return DistanceMatrix(labels=names, matrix=sub_matrix, field_set=self.field_set)

    def __repr__(self) -> str:
        return (
            f"DistanceMatrix(n={self.n}, "
            f"min={self.stats['min']:.4f}, "
            f"max={self.stats['max']:.4f}, "
            f"mean={self.stats['mean']:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ClusterAssignment  (per-document record)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusterAssignment:
    """Stores one document's cluster membership details."""

    label: str                  # Document / country name
    cluster_id: int             # Which cluster this document belongs to
    medoid: str                 # Name of the representative medoid
    distance_to_medoid: float   # How far this document is from its medoid

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "cluster_id": self.cluster_id,
            "medoid": self.medoid,
            "distance_to_medoid": round(self.distance_to_medoid, 6),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ClusterResult  (full clustering output)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusterResult:
    """
    Uniform result object returned by every clustering algorithm.

    Both HAC and K-Medoids produce a ClusterResult, so the GUI / reporting
    layer only needs to understand one type.

    Attributes
    ----------
    algorithm         : "hierarchical" | "kmedoids"
    n_clusters        : number of clusters produced
    assignments       : per-document assignments (label, cluster_id, medoid, dist)
    clusters          : cluster_id → list of document names
    medoids           : cluster_id → medoid name
    linkage           : linkage method used (HAC only, else None)
    inertia           : total sum of distances to assigned medoids
    silhouette_score  : mean silhouette over all documents (set by evaluation.py)
    cluster_stats     : per-cluster statistics computed by evaluation.py
    metadata          : arbitrary extra info (convergence iters, run time, etc.)
    """

    algorithm: str
    n_clusters: int
    assignments: List[ClusterAssignment]
    clusters: Dict[int, List[str]]          # cluster_id → member names
    medoids: Dict[int, str]                 # cluster_id → medoid name
    linkage: Optional[str] = None
    inertia: Optional[float] = None
    silhouette_score: Optional[float] = None
    cluster_stats: Dict[int, dict] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Convenience queries ───────────────────────────────────────────────────

    def cluster_of(self, label: str) -> int:
        """Return the cluster id for a document by name."""
        for a in self.assignments:
            if a.label == label:
                return a.cluster_id
        raise KeyError(f"{label!r} not found in assignments")

    def members_of(self, cluster_id: int) -> List[str]:
        """Return all document names in a cluster."""
        return self.clusters.get(cluster_id, [])

    def medoid_of(self, cluster_id: int) -> str:
        """Return the medoid name for a cluster."""
        return self.medoids[cluster_id]

    # ── Display ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable one-page summary of clustering results."""
        lines = [
            "",
            "=" * 62,
            f"  CLUSTERING RESULT — {self.algorithm.upper()}",
            "=" * 62,
            f"  Clusters     : {self.n_clusters}",
            f"  Documents    : {len(self.assignments)}",
        ]
        if self.linkage:
            lines.append(f"  Linkage      : {self.linkage}")
        if self.inertia is not None:
            lines.append(f"  Inertia      : {self.inertia:.4f}")
        if self.silhouette_score is not None:
            lines.append(f"  Silhouette   : {self.silhouette_score:.4f}")
        lines.append("")

        for cid in sorted(self.clusters):
            members = self.clusters[cid]
            medoid  = self.medoids.get(cid, "?")
            stats   = self.cluster_stats.get(cid, {})
            lines.append(f"  Cluster {cid}  ({len(members)} members)  medoid: {medoid}")
            lines.append(f"    {'-' * 56}")
            for name in sorted(members):
                marker = " *" if name == medoid else "  "
                a = next((x for x in self.assignments if x.label == name), None)
                dist_str = f"{a.distance_to_medoid:.4f}" if a else "?"
                lines.append(f"  {marker} {name:<35}  dist={dist_str}")
            if stats:
                lines.append(
                    f"    avg_to_medoid={stats.get('avg_distance_to_medoid', 0):.4f}  "
                    f"max_to_medoid={stats.get('max_distance_to_medoid', 0):.4f}  "
                    f"diameter={stats.get('diameter', 0):.4f}"
                )
                cp = stats.get("closest_pair")
                fp = stats.get("farthest_pair")
                if cp:
                    lines.append(
                        f"    closest:  {cp['a']} <-> {cp['b']}  (d={cp['distance']:.4f})"
                    )
                if fp:
                    lines.append(
                        f"    farthest: {fp['a']} <-> {fp['b']}  (d={fp['distance']:.4f})"
                    )
            lines.append("")

        lines.append("=" * 62)
        return "\n".join(lines)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "algorithm":       self.algorithm,
            "n_clusters":      self.n_clusters,
            "linkage":         self.linkage,
            "inertia":         round(self.inertia, 6) if self.inertia is not None else None,
            "silhouette_score": round(self.silhouette_score, 6) if self.silhouette_score is not None else None,
            "assignments":     [a.to_dict() for a in self.assignments],
            "clusters":        {str(k): v for k, v in self.clusters.items()},
            "medoids":         {str(k): v for k, v in self.medoids.items()},
            "cluster_stats":   {str(k): v for k, v in self.cluster_stats.items()},
            "metadata":        self.metadata,
        }

    def save(self, path: str) -> None:
        save_json(self.to_dict(), path)
        logger.info("ClusterResult saved -> %s", path)

    @classmethod
    def load(cls, path: str) -> "ClusterResult":
        """Restore a previously saved ClusterResult (for GUI consumption)."""
        data = load_json(path)
        assignments = [
            ClusterAssignment(**a) for a in data["assignments"]
        ]
        clusters = {int(k): v for k, v in data["clusters"].items()}
        medoids  = {int(k): v for k, v in data["medoids"].items()}
        cluster_stats = {int(k): v for k, v in data.get("cluster_stats", {}).items()}
        return cls(
            algorithm=data["algorithm"],
            n_clusters=data["n_clusters"],
            assignments=assignments,
            clusters=clusters,
            medoids=medoids,
            linkage=data.get("linkage"),
            inertia=data.get("inertia"),
            silhouette_score=data.get("silhouette_score"),
            cluster_stats=cluster_stats,
            metadata=data.get("metadata", {}),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ClusterConfig  (algorithm parameters)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusterConfig:
    """
    All tunable parameters for a clustering run.

    Pass one instance to ClusteringPipeline.run() instead of scattering
    keyword arguments.
    """

    # ── General ───────────────────────────────────────────────────────────────
    algorithm: str = "kmedoids"
    # Number of clusters to produce
    n_clusters: int = 5
    # Use semantic (weighted) TED or raw unit-cost TED
    use_semantic: bool = True

    # ── Hierarchical-specific ──────────────────────────────────────────────────
    # Linkage criterion: "single" | "complete" | "average"
    # Ward's linkage is NOT valid here because TED is not a Euclidean distance.
    linkage: str = "average"

    # ── K-Medoids-specific ────────────────────────────────────────────────────
    # Maximum PAM SWAP iterations
    max_iter: int = 100
    # "random"    — pick k random initial medoids
    # "heuristic" — greedy furthest-first initialization (more stable)
    init_method: str = "heuristic"
    # Reproducibility seed (used for random init only)
    random_seed: Optional[int] = 42

    def to_dict(self) -> dict:
        return {
            "algorithm":   self.algorithm,
            "n_clusters":  self.n_clusters,
            "use_semantic": self.use_semantic,
            "linkage":     self.linkage,
            "max_iter":    self.max_iter,
            "init_method": self.init_method,
            "random_seed": self.random_seed,
        }
