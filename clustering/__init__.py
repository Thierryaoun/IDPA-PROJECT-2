"""
clustering/
-----------
Project 2 — semi-structured document clustering using TED similarity.

Public API
----------
The simplest way to use this package:

    from clustering import ClusteringPipeline

    pipeline = ClusteringPipeline()
    pipeline.load_trees()
    pipeline.compute_distances()

    result = pipeline.run("kmedoids", n_clusters=5)
    pipeline.describe(result)
    pipeline.save_results(result)

For GUI / frontend integration, load a previously saved result without
re-running the pipeline:

    from clustering import ClusterResult
    result = ClusterResult.load("output/clustering/kmedoids_k5.json")

Individual components are also importable directly for custom use:

    from clustering.distance_matrix import load_country_trees, build_distance_matrix
    from clustering.hierarchical   import hierarchical_clustering
    from clustering.kmedoids       import kmedoids
    from clustering.evaluation     import attach_evaluation, silhouette_score
    from clustering.models         import DistanceMatrix, ClusterResult, ClusterConfig
"""

from .pipeline        import ClusteringPipeline
from .models          import DistanceMatrix, ClusterResult, ClusterAssignment, ClusterConfig
from .distance_matrix import load_country_trees, build_distance_matrix
from .hierarchical    import hierarchical_clustering
from .kmedoids        import kmedoids
from .evaluation      import attach_evaluation, silhouette_score
from .utils           import setup_logging, get_logger

__all__ = [
    # High-level
    "ClusteringPipeline",
    # Models
    "DistanceMatrix",
    "ClusterResult",
    "ClusterAssignment",
    "ClusterConfig",
    # Algorithms
    "hierarchical_clustering",
    "kmedoids",
    # Distance
    "load_country_trees",
    "build_distance_matrix",
    # Evaluation
    "attach_evaluation",
    "silhouette_score",
    # Utilities
    "setup_logging",
    "get_logger",
]

__version__ = "2.0.0"
