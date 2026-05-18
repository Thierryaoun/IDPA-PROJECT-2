"""
clustering/pipeline.py
----------------------
ClusteringPipeline — the single orchestrator for all clustering work.

Responsibilities
----------------
1. Load preprocessed country trees from disk.
2. Build (or restore from cache) the pairwise TED distance matrix.
3. Run a chosen clustering algorithm with a given config.
4. Attach evaluation metrics (silhouette, inertia, intra-cluster stats).
5. Print formatted summaries.
6. Save results to an output directory.

GUI integration note
--------------------
The pipeline intentionally returns plain ClusterResult objects (JSON-
serialisable data classes).  A GUI layer can:
  a) Call pipeline.run() directly and render the ClusterResult, or
  b) Load a pre-saved ClusterResult from disk without importing this module.
Either path requires no changes to this file.
"""

import os
import time
from typing import Dict, List, Optional

from .models import ClusterConfig, ClusterResult, DistanceMatrix
from .distance_matrix import build_distance_matrix, load_country_trees
from .hierarchical import hierarchical_clustering
from .kmedoids import kmedoids
from .evaluation import attach_evaluation
from .utils import get_logger, save_json, setup_logging

logger = get_logger(__name__)

# Default paths (relative to project root)
_DEFAULT_TREE_DIR   = "preprocessed_trees"
_DEFAULT_OUTPUT_DIR = "output/clustering"
_DEFAULT_CACHE_FILE = "output/clustering/distance_matrix.json"


class ClusteringPipeline:
    """
    End-to-end clustering pipeline over a directory of preprocessed trees.

    Typical usage::

        pipeline = ClusteringPipeline()
        pipeline.load_trees()
        pipeline.compute_distances()          # cached after first run

        result_km = pipeline.run("kmedoids",      n_clusters=5)
        result_hc = pipeline.run("hierarchical",  n_clusters=5, linkage="average")

        pipeline.describe(result_km)
        pipeline.save_results(result_km)
    """

    def __init__(
        self,
        tree_dir: str = _DEFAULT_TREE_DIR,
        output_dir: str = _DEFAULT_OUTPUT_DIR,
        cache_matrix: bool = True,
    ) -> None:
        self.tree_dir   = tree_dir
        self.output_dir = output_dir
        self.cache_matrix = cache_matrix

        self._trees: Dict           = {}
        self._dm: Optional[DistanceMatrix] = None

        os.makedirs(self.output_dir, exist_ok=True)

    # ── Step 1: Load trees ────────────────────────────────────────────────────

    def load_trees(
        self,
        country_filter: Optional[List[str]] = None,
    ) -> "ClusteringPipeline":
        """
        Load preprocessed country trees from *tree_dir*.

        Parameters
        ----------
        country_filter : if given, only load these country names
                         (useful for quick tests on a subset)

        Returns self for fluent chaining.
        """
        self._trees = load_country_trees(self.tree_dir, country_filter)
        logger.info("%d trees loaded.", len(self._trees))
        return self

    # ── Step 2: Distance matrix ───────────────────────────────────────────────

    def compute_distances(
        self,
        use_semantic: bool = True,
        force_recompute: bool = False,
    ) -> DistanceMatrix:
        """
        Build the pairwise TED distance matrix.

        The result is cached to disk (output_dir/distance_matrix.json) so
        subsequent calls are near-instant.  Pass force_recompute=True to
        ignore the cache and recompute from scratch.

        Returns the DistanceMatrix (also stored as self._dm).
        """
        if not self._trees:
            raise RuntimeError("Call load_trees() before compute_distances().")

        cache_path = (
            os.path.join(self.output_dir, "distance_matrix.json")
            if self.cache_matrix else None
        )

        if force_recompute and cache_path and os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info("Deleted cached matrix (force_recompute=True).")

        self._dm = build_distance_matrix(
            trees=self._trees,
            use_semantic=use_semantic,
            cache_path=cache_path,
        )
        return self._dm

    # ── Step 3: Run a clustering algorithm ───────────────────────────────────

    def run(
        self,
        algorithm: str = "kmedoids",
        n_clusters: int = 5,
        evaluate: bool = True,
        **kwargs,
    ) -> ClusterResult:
        """
        Run one clustering algorithm and return a ClusterResult.

        Parameters
        ----------
        algorithm  : "kmedoids" | "hierarchical"
        n_clusters : desired number of clusters
        evaluate   : if True, compute silhouette score and per-cluster stats
        **kwargs   : extra arguments forwarded to the algorithm
                     (e.g., linkage="complete" for hierarchical;
                            init_method="random", seed=0 for kmedoids)

        Returns
        -------
        ClusterResult  (with evaluation metrics if evaluate=True)
        """
        if self._dm is None:
            raise RuntimeError("Call compute_distances() before run().")

        t0 = time.perf_counter()

        if algorithm == "kmedoids":
            result = kmedoids(
                dm=self._dm,
                n_clusters=n_clusters,
                max_iter=kwargs.get("max_iter", 100),
                init_method=kwargs.get("init_method", "heuristic"),
                seed=kwargs.get("seed", 42),
                n_init=kwargs.get("n_init", 5),
            )
        elif algorithm == "hierarchical":
            result = hierarchical_clustering(
                dm=self._dm,
                n_clusters=n_clusters,
                linkage=kwargs.get("linkage", "average"),
            )
        else:
            raise ValueError(
                f"Unknown algorithm {algorithm!r}. "
                "Choose 'kmedoids' or 'hierarchical'."
            )

        elapsed = time.perf_counter() - t0
        result.metadata["run_time_seconds"] = round(elapsed, 3)
        result.metadata["n_documents"] = self._dm.n

        if evaluate:
            attach_evaluation(self._dm, result)

        logger.info(
            "run('%s', k=%d) finished in %.2fs  "
            "inertia=%.4f  silhouette=%s",
            algorithm, n_clusters, elapsed,
            result.inertia or 0,
            f"{result.silhouette_score:.4f}" if result.silhouette_score is not None else "n/a",
        )
        return result

    # ── Convenience: run both algorithms ─────────────────────────────────────

    def run_all(
        self,
        n_clusters: int = 5,
        hac_linkage: str = "average",
        km_init: str = "heuristic",
    ) -> Dict[str, ClusterResult]:
        """
        Run both K-Medoids and HAC, return a dict keyed by algorithm name.
        """
        return {
            "kmedoids": self.run(
                "kmedoids", n_clusters=n_clusters, init_method=km_init
            ),
            "hierarchical": self.run(
                "hierarchical", n_clusters=n_clusters, linkage=hac_linkage
            ),
        }

    # ── Step 4: Print summary ─────────────────────────────────────────────────

    def describe(self, result: ClusterResult) -> None:
        """Print a formatted summary of *result* to stdout."""
        print(result.summary())

    # ── Step 5: Save results ──────────────────────────────────────────────────

    def save_results(self, result: ClusterResult, tag: str = "") -> str:
        """
        Save a ClusterResult to output_dir as JSON.

        Returns the path of the saved file.
        """
        suffix = f"_{tag}" if tag else ""
        filename = f"{result.algorithm}_k{result.n_clusters}{suffix}.json"
        path = os.path.join(self.output_dir, filename)
        result.save(path)
        return path

    # ── Comparison helper ─────────────────────────────────────────────────────

    def compare_results(
        self,
        results: Dict[str, ClusterResult],
    ) -> None:
        """
        Print a side-by-side comparison table for multiple ClusterResult
        objects (e.g., the output of run_all()).
        """
        print()
        print("=" * 62)
        print("  ALGORITHM COMPARISON")
        print("=" * 62)
        header = f"  {'Algorithm':<20} {'k':>4} {'Inertia':>10} {'Silhouette':>12}"
        print(header)
        print("  " + "-" * 58)
        for name, r in results.items():
            sil = f"{r.silhouette_score:.4f}" if r.silhouette_score is not None else "n/a"
            inert = f"{r.inertia:.4f}" if r.inertia is not None else "n/a"
            label = name
            if r.linkage:
                label += f" ({r.linkage})"
            print(f"  {label:<20} {r.n_clusters:>4} {inert:>10} {sil:>12}")
        print("=" * 62)
        print()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def trees(self) -> Dict:
        return self._trees

    @property
    def distance_matrix(self) -> Optional[DistanceMatrix]:
        return self._dm

    @property
    def n_documents(self) -> int:
        return len(self._trees)
