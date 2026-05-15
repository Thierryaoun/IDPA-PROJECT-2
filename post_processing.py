import json
import os
import sys

from ted_algorithm import (
    TreeNode, load_tree_from_json, compute_edit_script,
    tree_edit_distance, tree_similarity, print_tree
)
from tree_patching import apply_patch, verify_patch


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Tree → JSON dictionary
# ═══════════════════════════════════════════════════════════════════════════════

def tree_to_dict(node: TreeNode) -> dict:
    
    if node.is_leaf():
        return {"label": node.label}
    return {
        "label": node.label,
        "children": [tree_to_dict(child) for child in node.children]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Tree → readable infobox text
# ═══════════════════════════════════════════════════════════════════════════════

def _label_to_display(label: str) -> str:
    
    return label.replace("_", " ").title()


def _value_to_display(label: str) -> str:
    # If the value looks numeric (starts with digit), keep as-is
    stripped = label.strip()
    if stripped and (stripped[0].isdigit() or stripped.startswith("-")):
        return stripped

    # Otherwise title-case for proper nouns
    return stripped.title()


def tree_to_infobox_text(tree: TreeNode) -> str:

    # Collect all field → value entries by walking the tree
    entries = []
    _collect_entries(tree, "", entries)

    if not entries:
        return "(empty tree)"

    # Find the infobox title (official name) for the header
    title = ""
    for field, value in entries:
        if field.lower() in ("infobox title", "infobox_title"):
            title = value
            break
    if not title:
        # Fallback: use the "name" field
        for field, value in entries:
            if field.lower() == "name":
                title = value
                break

    # Calculate column widths
    max_field_len = max(len(field) for field, _ in entries)
    max_value_len = max(len(value) for _, value in entries)
    inner_width = max(max_field_len + max_value_len + 3, len(title) + 4, 40)

    # Build the output
    lines = []

    # Header box
    lines.append("╔" + "═" * (inner_width + 2) + "╗")
    lines.append("║ " + title.upper().center(inner_width) + " ║")
    lines.append("╠" + "═" * (inner_width + 2) + "╣")

    # Field entries
    for field, value in entries:
        # Skip the infobox_title from the body (already in header)
        if field.lower() in ("infobox title", "infobox_title"):
            continue

        entry = f" {field:<{max_field_len}} : {value}"
        # Pad to inner width
        entry = entry.ljust(inner_width)
        lines.append("║ " + entry + " ║")

    # Footer
    lines.append("╚" + "═" * (inner_width + 2) + "╝")

    return "\n".join(lines)


def _collect_entries(node: TreeNode, prefix: str, entries: list):

    for child in node.children:
        if child.is_leaf():
            # This is a content node directly under a structural parent
            # The parent's label is the field name
            field_name = prefix if prefix else _label_to_display(node.label)
            value = _value_to_display(child.label)
            entries.append((field_name, value))
        else:
            # This is a nested structural node
            # Build the display name by combining prefix + this label
            display = _label_to_display(child.label)
            if prefix:
                nested_prefix = f"{prefix} {display}"
            else:
                nested_prefix = display

            # Check if this structural node has only leaf children
            # (simple field with value)
            if all(c.is_leaf() for c in child.children):
                for leaf in child.children:
                    value = _value_to_display(leaf.label)
                    entries.append((nested_prefix, value))
            else:
                # Has nested structural children — recurse
                # But also collect any direct leaf children first
                has_leaves = any(c.is_leaf() for c in child.children)
                has_struct = any(c.is_structural() for c in child.children)

                if has_leaves and has_struct:
                    # Mixed: collect leaves directly, recurse into structural
                    for c in child.children:
                        if c.is_leaf():
                            entries.append((nested_prefix, _value_to_display(c.label)))
                        else:
                            _collect_entries(child, nested_prefix, entries)
                    # Avoid double-recursion: structural children already handled
                    pass
                else:
                    # All structural children — recurse
                    _collect_entries(child, nested_prefix, entries)


def tree_to_simple_text(tree: TreeNode) -> str:
    
    entries = []
    _collect_entries(tree, "", entries)

    lines = []
    for field, value in entries:
        lines.append(f"{field}: {value}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Export functions
# ═══════════════════════════════════════════════════════════════════════════════

def export_infobox(tree: TreeNode, filename: str, output_dir: str = "."):
    
    os.makedirs(output_dir, exist_ok=True)

    # Export JSON tree
    json_path = os.path.join(output_dir, f"{filename}_tree.json")
    tree_dict = tree_to_dict(tree)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tree_dict, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON tree   : {json_path}")

    # Export infobox text
    txt_path = os.path.join(output_dir, f"{filename}_infobox.txt")
    infobox_text = tree_to_infobox_text(tree)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(infobox_text)
    print(f"  Saved infobox text: {txt_path}")

    # Also export simple text format
    simple_path = os.path.join(output_dir, f"{filename}_simple.txt")
    simple_text = tree_to_simple_text(tree)
    with open(simple_path, "w", encoding="utf-8") as f:
        f.write(simple_text)
    print(f"  Saved simple text : {simple_path}")

    return json_path, txt_path, simple_path


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_post_processing(tree: TreeNode):
    """
    Verify post-processing by displaying all output formats.

    Steps:
      1. Convert tree to JSON dict and display
      2. Convert tree to infobox text and display
      3. Convert tree to simple text and display

    This allows visual inspection that the tree was correctly
    transformed back into readable form.
    """
    print("=" * 60)
    print("  POST-PROCESSING RESULT")
    print("=" * 60)
    print()

    # JSON representation
    print("  JSON TREE:")
    print("  " + "-" * 40)
    tree_dict = tree_to_dict(tree)
    json_str = json.dumps(tree_dict, indent=2, ensure_ascii=False)
    for line in json_str.split("\n"):
        print(f"  {line}")
    print()

    # Infobox format
    print("  INFOBOX FORMAT:")
    print("  " + "-" * 40)
    infobox = tree_to_infobox_text(tree)
    for line in infobox.split("\n"):
        print(f"  {line}")
    print()

    # Simple text format
    print("  SIMPLE TEXT FORMAT:")
    print("  " + "-" * 40)
    simple = tree_to_simple_text(tree)
    for line in simple.split("\n"):
        print(f"  {line}")
    print()

    # Statistics
    print("  TREE STATISTICS:")
    print("  " + "-" * 40)
    print(f"  Total nodes    : {tree.size()}")
    print(f"  Tree depth     : {tree.depth()}")
    print(f"  Leaf count     : {len(tree.leaves())}")
    structural = tree.size() - len(tree.leaves())
    print(f"  Structural     : {structural}")
    print(f"  Content (leaf) : {len(tree.leaves())}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Full pipeline demo
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(t1: TreeNode, t2: TreeNode,
                      name1: str, name2: str,
                      output_dir: str = "output"):
    """
    Run the complete pipeline from differencing through post-processing:

    1. Compute TED and similarity between T1 and T2
    2. Extract edit script (diff)
    3. Apply patch to T1 to produce patched tree
    4. Verify patch (TED between patched and T2 should be 0)
    5. Post-process: convert patched tree to infobox format
    6. Export all outputs

    This demonstrates the full project flow per Figure 3.
    """
    print("=" * 60)
    print("  FULL PIPELINE: Differencing → Patching → Post-Processing")
    print("=" * 60)
    print()

    # ── Step 1: Similarity ──
    print(f"  Step 1: Computing similarity ({name1} vs {name2})")
    print("  " + "-" * 50)
    sim = tree_similarity(t1, t2)
    print(f"    Structural similarity : {sim['structural_similarity_percent']:.2f}%")
    print(f"    Content similarity    : {sim['content_similarity_percent']:.2f}%")
    print(f"    Overall similarity    : {sim['overall_similarity_percent']:.2f}%")
    print(f"    Tree Edit Distance    : {sim['distance']}")
    print()

    # ── Step 2: Edit script ──
    print(f"  Step 2: Computing edit script (diff)")
    print("  " + "-" * 50)
    script = compute_edit_script(t1, t2)
    edits = [op for op in script if op["op"] != "Match"]
    matches = [op for op in script if op["op"] == "Match"]
    print(f"    {len(matches)} matches, {len(edits)} edits")
    print()

    # ── Step 3: Patching ──
    print(f"  Step 3: Applying patch ({name1} + diff → {name2})")
    print("  " + "-" * 50)
    patched = apply_patch(t1, script)
    print()

    # ── Step 4: Verify patch ──
    print(f"  Step 4: Verifying patch")
    print("  " + "-" * 50)
    dist = tree_edit_distance(patched, t2)
    if dist == 0:
        print(f"    TED(patched, target) = {dist}")
        print(f"    ✓ Patch successful: patched tree equals target tree")
    else:
        print(f"    TED(patched, target) = {dist}")
        print(f"    ✗ Patch incomplete: residual distance = {dist}")
    print()

    # ── Step 5: Post-processing ──
    print(f"  Step 5: Post-processing patched tree")
    print("  " + "-" * 50)
    print()

    # Show source infobox
    print(f"  SOURCE ({name1}):")
    source_infobox = tree_to_infobox_text(t1)
    for line in source_infobox.split("\n"):
        print(f"    {line}")
    print()

    # Show target infobox
    print(f"  TARGET ({name2}):")
    target_infobox = tree_to_infobox_text(t2)
    for line in target_infobox.split("\n"):
        print(f"    {line}")
    print()

    # Show patched infobox
    print(f"  PATCHED RESULT ({name1} → {name2}):")
    patched_infobox = tree_to_infobox_text(patched)
    for line in patched_infobox.split("\n"):
        print(f"    {line}")
    print()

    # ── Step 6: Export ──
    print(f"  Step 6: Exporting results")
    print("  " + "-" * 50)
    os.makedirs(output_dir, exist_ok=True)

    # Export source
    export_infobox(t1, f"source_{name1.replace(' ', '_')}", output_dir)

    # Export target
    export_infobox(t2, f"target_{name2.replace(' ', '_')}", output_dir)

    # Export patched
    export_infobox(patched, f"patched_{name1.replace(' ', '_')}_to_{name2.replace(' ', '_')}", output_dir)

    # Export diff
    diff_path = os.path.join(output_dir, f"diff_{name1.replace(' ', '_')}_to_{name2.replace(' ', '_')}.json")
    diff_data = {
        "source": name1,
        "target": name2,
        "similarity": {
            "structural": sim["structural_similarity_percent"],
            "content": sim["content_similarity_percent"],
            "overall": sim["overall_similarity_percent"],
        },
        "tree_edit_distance": sim["distance"],
        "edit_script": script
    }
    with open(diff_path, "w", encoding="utf-8") as f:
        json.dump(diff_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved diff file   : {diff_path}")

    print()
    print("  ★ Pipeline complete")
    print()

    return patched


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Demo with sample data
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """Demonstrate post-processing with sample Lebanon → Syria pipeline."""

    # Build sample trees
    t1 = TreeNode("country", [
        TreeNode("name", [TreeNode("lebanon")]),
        TreeNode("infobox_title", [TreeNode("lebanese republic")]),
        TreeNode("capital_and_largest_city", [TreeNode("beirut")]),
        TreeNode("official_languages", [TreeNode("arabic")]),
        TreeNode("president", [TreeNode("joseph aoun")]),
        TreeNode("prime_minister", [TreeNode("nawaf salam")]),
        TreeNode("legislature", [TreeNode("parliament")]),
        TreeNode("area", [
            TreeNode("total", [TreeNode("10,452 km2")]),
            TreeNode("water_percent", [TreeNode("1.8")])
        ]),
        TreeNode("population", [
            TreeNode("total", [TreeNode("5,364,482")]),
            TreeNode("density", [TreeNode("513/km2")])
        ])
    ])

    t2 = TreeNode("country", [
        TreeNode("name", [TreeNode("syria")]),
        TreeNode("infobox_title", [TreeNode("syrian arab republic")]),
        TreeNode("capital_and_largest_city", [TreeNode("damascus")]),
        TreeNode("official_languages", [TreeNode("arabic")]),
        TreeNode("president", [TreeNode("ahmed al-sharaa")]),
        TreeNode("prime_minister", [TreeNode("mohammed al-bashir")]),
        TreeNode("legislature", [TreeNode("people's assembly")]),
        TreeNode("area", [
            TreeNode("total", [TreeNode("185,180 km2")]),
            TreeNode("water_percent", [TreeNode("1.1")])
        ]),
        TreeNode("population", [
            TreeNode("total", [TreeNode("26,019,711")]),
            TreeNode("density", [TreeNode("99/km2")])
        ])
    ])

    # Run the full pipeline
    run_full_pipeline(t1, t2, "Lebanon", "Syria", output_dir="output")

    # Also show verification for the patched tree
    print()
    script = compute_edit_script(t1, t2)
    patched = apply_patch(t1, script)
    verify_post_processing(patched)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Main — real data or demo
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    TREE_DIR = "preprocessed_trees"

    if os.path.isdir(TREE_DIR):
        files = sorted(f for f in os.listdir(TREE_DIR)
                       if f.endswith(".json") and not f.startswith("_"))

        if len(files) >= 2:
            countries = {}
            for f in files:
                countries[f.replace(".json", "").replace("_", " ")] = \
                    os.path.join(TREE_DIR, f)
            names = list(countries.keys())

            print("=" * 60)
            print("  POST-PROCESSING — Full Pipeline")
            print("=" * 60)
            print(f"\n  {len(names)} countries available.")
            for i, n in enumerate(names, 1):
                print(f"    {i:3d}. {n}")
            print("\n  Pick two (e.g. '1 2'), or Enter for first two:\n")

            try:
                inp = input("  > ").strip()
            except EOFError:
                inp = ""

            if inp and len(inp.split()) >= 2:
                try:
                    a, b = int(inp.split()[0]) - 1, int(inp.split()[1]) - 1
                    n1, n2 = names[a], names[b]
                except (ValueError, IndexError):
                    n1, n2 = names[0], names[1]
            else:
                n1, n2 = names[0], names[1]

            t1 = load_tree_from_json(countries[n1])
            t2 = load_tree_from_json(countries[n2])

            run_full_pipeline(t1, t2, n1, n2, output_dir="output")
            return

    # Fallback to demo
    demo()


if __name__ == "__main__":
    main()
