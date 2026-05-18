"""
run_clustering.py
-----------------
Entry point for Project 2 — document clustering via TED similarity.

Usage
-----
Run with no arguments for the full pipeline on all countries (slow first
run, fast on subsequent runs thanks to the cached distance matrix):

    python run_clustering.py

Pass --demo to run a fast smoke-test on a small hand-picked subset instead:

    python run_clustering.py --demo

Pass --validate to run unit tests on the core clustering logic without
touching any real country data:

    python run_clustering.py --validate
"""

import os
import sys

# ── Make sure the project root is on the path ────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from clustering import (
    ClusteringPipeline,
    ClusterResult,
    DistanceMatrix,
    ClusterConfig,
    setup_logging,
    hierarchical_clustering,
    kmedoids,
    attach_evaluation,
)

# ─────────────────────────────────────────────────────────────────────────────
# Validation (unit tests — no real data needed)
# ─────────────────────────────────────────────────────────────────────────────

def run_validation() -> bool:
    """
    Test the clustering algorithms on a tiny synthetic distance matrix.
    Returns True if all assertions pass.
    """
    print("=" * 62)
    print("  VALIDATION — synthetic distance matrix")
    print("=" * 62)
    results = []

    # ── Build a 6-point distance matrix with 2 obvious clusters ──────────────
    # Points 0,1,2 are very close (cluster A); points 3,4,5 are very close
    # (cluster B); the two clusters are far apart.
    labels = ["A0", "A1", "A2", "B3", "B4", "B5"]
    m = [
        [0.00, 0.05, 0.07, 0.90, 0.92, 0.88],
        [0.05, 0.00, 0.06, 0.91, 0.89, 0.93],
        [0.07, 0.06, 0.00, 0.88, 0.90, 0.91],
        [0.90, 0.91, 0.88, 0.00, 0.04, 0.06],
        [0.92, 0.89, 0.90, 0.04, 0.00, 0.05],
        [0.88, 0.93, 0.91, 0.06, 0.05, 0.00],
    ]
    dm = DistanceMatrix(labels=labels, matrix=m)

    # ── Test 1: K-Medoids k=2 should separate A from B ───────────────────────
    r_km = kmedoids(dm, n_clusters=2, init_method="heuristic")
    attach_evaluation(dm, r_km)
    a_ids = {r_km.cluster_of(lbl) for lbl in ["A0", "A1", "A2"]}
    b_ids = {r_km.cluster_of(lbl) for lbl in ["B3", "B4", "B5"]}
    ok = (len(a_ids) == 1) and (len(b_ids) == 1) and (a_ids != b_ids)
    results.append(ok)
    print(f"\n  1. K-Medoids k=2 separates clusters:   {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"     A-ids={a_ids}  B-ids={b_ids}")

    # ── Test 2: Silhouette should be > 0.5 for obvious clusters ──────────────
    ok = r_km.silhouette_score is not None and r_km.silhouette_score > 0.5
    results.append(ok)
    print(f"  2. Silhouette > 0.5:                    {'PASS' if ok else 'FAIL'}"
          f"  (score={r_km.silhouette_score:.4f})")

    # ── Test 3: Inertia should be small ──────────────────────────────────────
    ok = r_km.inertia is not None and r_km.inertia < 0.5
    results.append(ok)
    print(f"  3. K-Medoids inertia < 0.5:             {'PASS' if ok else 'FAIL'}"
          f"  (inertia={r_km.inertia:.4f})")

    # ── Test 4: HAC average linkage k=2 ──────────────────────────────────────
    r_hac = hierarchical_clustering(dm, n_clusters=2, linkage="average")
    attach_evaluation(dm, r_hac)
    a_ids = {r_hac.cluster_of(lbl) for lbl in ["A0", "A1", "A2"]}
    b_ids = {r_hac.cluster_of(lbl) for lbl in ["B3", "B4", "B5"]}
    ok = (len(a_ids) == 1) and (len(b_ids) == 1) and (a_ids != b_ids)
    results.append(ok)
    print(f"  4. HAC average k=2 separates clusters:  {'PASS' if ok else 'FAIL'}")

    # ── Test 5: HAC silhouette ────────────────────────────────────────────────
    ok = r_hac.silhouette_score is not None and r_hac.silhouette_score > 0.5
    results.append(ok)
    print(f"  5. HAC silhouette > 0.5:                {'PASS' if ok else 'FAIL'}"
          f"  (score={r_hac.silhouette_score:.4f})")

    # ── Test 6: DistanceMatrix save / load round-trip ─────────────────────────
    tmp_path = os.path.join("output", "clustering", "_test_matrix.json")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    dm.save(tmp_path)
    dm2 = DistanceMatrix.load(tmp_path)
    ok = (dm2.labels == dm.labels and
          all(abs(dm2.matrix[i][j] - dm.matrix[i][j]) < 1e-9
              for i in range(dm.n) for j in range(dm.n)))
    results.append(ok)
    print(f"  6. DistanceMatrix save/load round-trip: {'PASS' if ok else 'FAIL'}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # ── Test 7: ClusterResult save / load round-trip ─────────────────────────
    tmp_path = os.path.join("output", "clustering", "_test_result.json")
    r_km.save(tmp_path)
    r2 = ClusterResult.load(tmp_path)
    ok = (r2.algorithm == "kmedoids" and
          r2.n_clusters == 2 and
          len(r2.assignments) == 6)
    results.append(ok)
    print(f"  7. ClusterResult save/load round-trip:  {'PASS' if ok else 'FAIL'}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # ── Test 8: HAC merge history is stored in metadata ──────────────────────
    ok = "merge_history" in r_hac.metadata and len(r_hac.metadata["merge_history"]) == 4
    results.append(ok)
    print(f"  8. HAC merge history (4 merges for k=2):{' PASS' if ok else ' FAIL'}"
          f"  (merges={len(r_hac.metadata.get('merge_history',[]))})")

    # ── Test 9: K-Medoids convergence metadata ───────────────────────────────
    ok = r_km.metadata.get("converged") is True
    results.append(ok)
    print(f"  9. K-Medoids converged flag set:        {'PASS' if ok else 'FAIL'}")

    # ── Test 10: k=1 edge case ───────────────────────────────────────────────
    r_k1 = kmedoids(dm, n_clusters=1)
    ok = r_k1.n_clusters == 1 and len(r_k1.clusters[0]) == 6
    results.append(ok)
    print(f"  10. k=1 puts all docs in one cluster:   {'PASS' if ok else 'FAIL'}")

    passed = sum(results)
    total  = len(results)
    print()
    print(f"  {passed}/{total} tests passed")
    print("=" * 62)
    return passed == total


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode — small subset of real countries
# ─────────────────────────────────────────────────────────────────────────────

# A hand-picked set covering different world regions for a visible demo.
DEMO_COUNTRIES = [
    "Lebanon", "Syria", "Jordan", "Iraq", "Egypt",          # Middle East / Levant
    "France", "Germany", "Switzerland", "Italy", "Spain",   # Western Europe
    "Brazil", "Argentina", "Chile", "Colombia", "Peru",     # Latin America
    "China", "Japan", "India", "South Korea", "Indonesia",  # East / South Asia
    "Nigeria", "Kenya", "Ethiopia", "South Africa", "Ghana",# Africa
]


def run_demo() -> None:
    """
    Fast demonstration on DEMO_COUNTRIES (25 countries, 300 pairs).

    This is the recommended first run to verify the system works before
    launching the full 193-country pipeline.
    """
    print()
    print("=" * 62)
    print("  PROJECT 2 — DEMO (25 countries)")
    print("=" * 62)

    pipeline = ClusteringPipeline(
        output_dir="output/clustering",
        cache_matrix=True,
    )

    # Load only the demo subset
    pipeline.load_trees(country_filter=DEMO_COUNTRIES)
    actual_countries = list(pipeline.trees.keys())
    print(f"\n  Loaded {len(actual_countries)} countries:")
    for c in sorted(actual_countries):
        print(f"    - {c}")

    if len(actual_countries) < 4:
        print("\n  [WARN] Too few countries found in preprocessed_trees/.")
        print("  Run preprocessing.py first to generate tree files.")
        return

    # Compute distance matrix (cached as output/clustering/distance_matrix_demo.json)
    demo_cache = "output/clustering/distance_matrix_demo.json"
    pipeline._dm = None  # reset so we can set a custom cache path
    from clustering.distance_matrix import build_distance_matrix
    pipeline._dm = build_distance_matrix(
        trees=pipeline.trees,
        use_semantic=True,
        cache_path=demo_cache,
    )

    print(f"\n  Distance matrix: {pipeline._dm}")

    # Run both algorithms with k=5
    print("\n  Running K-Medoids (k=5)…")
    r_km = pipeline.run("kmedoids", n_clusters=5)

    print("\n  Running Hierarchical (average linkage, k=5)…")
    r_hac = pipeline.run("hierarchical", n_clusters=5, linkage="average")

    # Print results
    pipeline.describe(r_km)
    pipeline.describe(r_hac)
    pipeline.compare_results({"kmedoids": r_km, "hierarchical (avg)": r_hac})

    # Save results
    km_path  = pipeline.save_results(r_km,  tag="demo")
    hac_path = pipeline.save_results(r_hac, tag="demo")
    print(f"  Results saved:")
    print(f"    {km_path}")
    print(f"    {hac_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline — all countries
# ─────────────────────────────────────────────────────────────────────────────

def run_full(n_clusters: int = 7) -> None:
    """
    Full pipeline over all countries in preprocessed_trees/.

    The distance matrix is cached so the first run is the only slow one.
    Subsequent runs load the matrix from disk and cluster in seconds.
    """
    print()
    print("=" * 62)
    print(f"  PROJECT 2 — FULL PIPELINE  (k={n_clusters})")
    print("=" * 62)

    pipeline = ClusteringPipeline(
        output_dir="output/clustering",
        cache_matrix=True,
    )

    pipeline.load_trees()

    if pipeline.n_documents < 2:
        print("  [ERROR] Not enough trees found in preprocessed_trees/.")
        print("  Run preprocessing.py first.")
        return

    print(f"\n  {pipeline.n_documents} countries loaded.")
    pipeline.compute_distances(use_semantic=True)
    print(f"  {pipeline.distance_matrix}")

    # Both algorithms
    results = pipeline.run_all(
        n_clusters=n_clusters,
        hac_linkage="average",
        km_init="heuristic",
    )

    for name, r in results.items():
        pipeline.describe(r)
        pipeline.save_results(r)

    pipeline.compare_results(results)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging(level="INFO", log_file="output/clustering/clustering.log")

    mode = sys.argv[1] if len(sys.argv) > 1 else "--demo"

    if mode == "--validate":
        ok = run_validation()
        sys.exit(0 if ok else 1)

    elif mode == "--demo":
        run_demo()

    elif mode == "--full":
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        run_full(n_clusters=k)

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
