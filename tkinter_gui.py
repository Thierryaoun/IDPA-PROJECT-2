import os
import tkinter as tk
from tkinter import ttk

from ted_algorithm import load_tree_from_json, compute_similarity


TREE_DIR = "preprocessed_trees"


class CountrySimilarityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Country Infobox TED Similarity")
        self.geometry("780x520")
        self.use_semantic = tk.BooleanVar(value=True)
        self.countries = self._load_country_files()
        self._build_ui()

    def _load_country_files(self):
        if not os.path.isdir(TREE_DIR):
            return []
        return sorted(
            f.replace(".json", "").replace("_", " ")
            for f in os.listdir(TREE_DIR)
            if f.endswith(".json") and not f.startswith("_")
        )

    def _build_ui(self):
        controls = ttk.Frame(self, padding=12)
        controls.pack(fill="x")

        ttk.Label(controls, text="Country 1").grid(row=0, column=0, sticky="w")
        ttk.Label(controls, text="Country 2").grid(row=0, column=1, sticky="w")

        self.country1 = ttk.Combobox(controls, values=self.countries, width=34)
        self.country2 = ttk.Combobox(controls, values=self.countries, width=34)
        self.country1.grid(row=1, column=0, padx=(0, 8), sticky="ew")
        self.country2.grid(row=1, column=1, padx=(0, 8), sticky="ew")
        if len(self.countries) >= 2:
            self.country1.set(self.countries[0])
            self.country2.set(self.countries[1])

        ttk.Checkbutton(
            controls,
            text="Use semantic preprocessing",
            variable=self.use_semantic,
        ).grid(row=2, column=0, sticky="w", pady=(10, 0))

        ttk.Button(controls, text="Compare", command=self.compare).grid(
            row=2, column=1, sticky="e", pady=(10, 0)
        )

        warning = (
            "This score measures normalized infobox tree similarity, "
            "not real-world geopolitical similarity."
        )
        ttk.Label(self, text=warning, foreground="#8a5a00", padding=(12, 4)).pack(
            fill="x"
        )

        self.output = tk.Text(self, wrap="word", height=22)
        self.output.pack(fill="both", expand=True, padx=12, pady=12)

    def _path_for(self, name):
        return os.path.join(TREE_DIR, f"{name.replace(' ', '_')}.json")

    def compare(self):
        self.output.delete("1.0", "end")
        name1, name2 = self.country1.get(), self.country2.get()
        if not name1 or not name2:
            self.output.insert("end", "Select two countries.\n")
            return

        t1 = load_tree_from_json(self._path_for(name1))
        t2 = load_tree_from_json(self._path_for(name2))
        result = compute_similarity(
            t1,
            t2,
            use_semantic_preprocessing=self.use_semantic.get(),
        )

        lines = [
            f"{name1} vs {name2}",
            "",
            f"TED distance: {result['raw_ted_distance']}",
            f"Raw normalized similarity: {result['raw_tree_similarity_percent']:.2f}%",
            f"Tree size before preprocessing: "
            f"{result['raw_tree1_size']} / {result['raw_tree2_size']}",
        ]

        if result.get("semantic_preprocessing"):
            lines.extend([
                f"Semantic TED distance: {result['semantic_ted_distance']}",
                f"Weighted semantic tree similarity: "
                f"{result['weighted_semantic_tree_similarity_percent']:.2f}%",
                f"Tree size after preprocessing: "
                f"{result['semantic_tree1_size']} / {result['semantic_tree2_size']}",
                f"Ignored field count: {result['ignored_field_count']}",
                "",
                "Weighted cost contribution by field/category:",
            ])
            for field, cost in result["debug"]["weighted_cost_contribution_by_field"].items():
                lines.append(f"  {field}: {cost}")
            lines.extend(["", "Top edit operations:"])
            for op in result["debug"]["top_edit_operations"][:10]:
                lines.append(f"  {op}")

        lines.extend(["", f"WARNING: {result['warning']}"])
        self.output.insert("end", "\n".join(lines))


if __name__ == "__main__":
    CountrySimilarityApp().mainloop()
