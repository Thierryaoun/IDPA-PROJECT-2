import copy
import json
import os
import sys

# Import from the existing TED module (same directory)
from ted_algorithm import (
    TreeNode, load_tree_from_json, load_tree_from_dict,
    compute_edit_script, tree_edit_distance, tree_similarity,
    print_tree, print_edit_script
)




def deep_copy_tree(node: TreeNode) -> TreeNode:
    """
    Create a deep copy of a tree so that patching does not modify the original.
    We avoid Python's copy.deepcopy because it copies id()-based references
    that could break the TED algorithm's path computation on the copy.
    Instead we rebuild the tree from scratch.
    """
    new = TreeNode(node.label)
    for child in node.children:
        new.add_child(deep_copy_tree(child))
    return new




def find_node_by_path(root: TreeNode, path: str) -> TreeNode:

    if not path:
        return root

    parts = path.split(".")
    current = root

    if parts[0] != current.label:
        return None

    for part in parts[1:]:
        found = None
        for child in current.children:
            if child.label == part:
                found = child
                break
        if found is None:
            return None
        current = found

    return current


def find_parent_by_path(root: TreeNode, path: str):

    if not path:
        return None, None

    parts = path.split(".")
    if len(parts) < 2:
        return root if root.label == parts[0] else None, parts[0]

    parent_path = ".".join(parts[:-1])
    parent = find_node_by_path(root, parent_path)
    return parent, parts[-1]




def update_node(root: TreeNode, path: str, old_label: str, new_label: str,
                node_type: str) -> bool:

    if node_type == "structural":
        # Find the structural node itself and rename it
        node = find_node_by_path(root, path)
        if node and node.label == old_label:
            node.label = new_label
            return True
        # Fallback: the path uses the old label, try parent + old child
        parent, _ = find_parent_by_path(root, path)
        if parent:
            for child in parent.children:
                if child.label == old_label and child.is_structural():
                    child.label = new_label
                    return True
        return False

    elif node_type == "content":
        # Content node: path points to the structural parent
        # Find the parent and rename the matching leaf child
        target = find_node_by_path(root, path)
        if target:
            for child in target.children:
                if child.is_leaf() and child.label == old_label:
                    child.label = new_label
                    return True
        return False

    return False


def delete_node(root: TreeNode, path: str, label: str, node_type: str) -> bool:

    if node_type == "structural":
        # Path is the node's own path; find parent and remove child
        parent, child_label = find_parent_by_path(root, path)
        if parent:
            for i, child in enumerate(parent.children):
                if child.label == label:
                    parent.children.pop(i)
                    return True
        return False

    elif node_type == "content":
        # Path points to the structural parent containing this leaf
        target = find_node_by_path(root, path)
        if target:
            for i, child in enumerate(target.children):
                if child.is_leaf() and child.label == label:
                    target.children.pop(i)
                    return True
        return False

    return False


def insert_node(root: TreeNode, path: str, label: str, node_type: str) -> bool:

    if node_type == "structural":
        # Path is the would-be path of the new node
        # The parent is everything except the last component
        parent_path = ".".join(path.split(".")[:-1]) if "." in path else ""
        parent = find_node_by_path(root, parent_path) if parent_path else root
        if parent:
            # Only add if not already present
            existing = [c for c in parent.children if c.label == label]
            if not existing:
                parent.add_child(TreeNode(label))
            return True
        return False

    elif node_type == "content":
        # Path points to the structural parent
        target = find_node_by_path(root, path)
        if target:
            target.add_child(TreeNode(label))
            return True
        # If the structural parent doesn't exist yet, try to create it
        # (it may have been inserted in a previous operation)
        return False

    return False




def apply_patch(tree: TreeNode, edit_script: list) -> TreeNode:

    patched = deep_copy_tree(tree)

    # Classify operations
    updates = [op for op in edit_script if op["op"] == "Update"]
    deletes = [op for op in edit_script if op["op"] == "Delete"]
    inserts = [op for op in edit_script if op["op"] == "Insert"]

    struct_deletes = [op for op in deletes if op["type"] == "structural"]
    content_deletes = [op for op in deletes if op["type"] == "content"]
    struct_updates = [op for op in updates if op["type"] == "structural"]
    content_updates = [op for op in updates if op["type"] == "content"]
    struct_inserts = [op for op in inserts if op["type"] == "structural"]
    content_inserts = [op for op in inserts if op["type"] == "content"]

    applied = {"ok": 0, "skip": 0}

    def _try(fn, *args):
        if fn(*args):
            applied["ok"] += 1
        else:
            applied["skip"] += 1

    # Phase 1: DELETE structural (deepest paths first → reverse by path depth)
    for op in sorted(struct_deletes,
                     key=lambda o: o.get("path", "").count("."), reverse=True):
        _try(delete_node, patched, op.get("path", ""), op["node1"], "structural")

    # Phase 2: DELETE content
    for op in reversed(content_deletes):
        _try(delete_node, patched, op.get("path", ""), op["node1"], "content")

    # Phase 3: UPDATE structural then content
    for op in struct_updates:
        _try(update_node, patched, op.get("path", ""),
             op["node1"], op["node2"], "structural")
    for op in content_updates:
        _try(update_node, patched, op.get("path", ""),
             op["node1"], op["node2"], "content")

    # Phase 4: INSERT structural (shallowest paths first → by path depth)
    for op in sorted(struct_inserts,
                     key=lambda o: o.get("path", "").count(".")):
        _try(insert_node, patched, op.get("path", ""), op["node2"], "structural")

    # Phase 5: INSERT content
    for op in content_inserts:
        _try(insert_node, patched, op.get("path", ""), op["node2"], "content")

    total = len(updates) + len(deletes) + len(inserts)
    print(f"    Patch applied: {applied['ok']}/{total} operations")
    print(f"      Deletes: structural={len(struct_deletes)}, content={len(content_deletes)}")
    print(f"      Updates: structural={len(struct_updates)}, content={len(content_updates)}")
    print(f"      Inserts: structural={len(struct_inserts)}, content={len(content_inserts)}")
    if applied["skip"] > 0:
        print(f"      Skipped: {applied['skip']} (already handled by subtree delete/insert)")

    # ── Phase 6: REORDER children to match the target tree ordering ──
    # The edit script operations are in postorder (leaves before parents).
    # We reconstruct the intended child order from the script: operations
    # on children of the same parent appear in left-to-right order.
    _reorder_from_script(patched, edit_script)

    return patched


def _reorder_from_script(root: TreeNode, edit_script: list):

    target_order = {}  # parent_path → [child_label, ...]

    for op in edit_script:
        path = op.get("path", "")
        if not path:
            continue

        # For a structural node at path "country.gdp", parent is "country"
        # For a content node at path "country.gdp", parent is "country.gdp"
        if op["type"] == "structural":
            parts = path.split(".")
            if len(parts) >= 2:
                parent_path = ".".join(parts[:-1])
                child_label = op.get("node2", op.get("node1", ""))
                if parent_path not in target_order:
                    target_order[parent_path] = []
                if child_label and child_label not in target_order[parent_path]:
                    target_order[parent_path].append(child_label)

    # Now reorder children of each node according to target_order
    _apply_order(root, root.label, target_order)


def _apply_order(node: TreeNode, current_path: str, target_order: dict):
    """Recursively reorder children based on target ordering."""
    if not node.children:
        return

    order = target_order.get(current_path, [])
    if order:
        # Sort children: those in `order` come first in that order,
        # then remaining children in their current order
        ordered = []
        remaining = list(node.children)

        for label in order:
            for i, child in enumerate(remaining):
                if child.label == label:
                    ordered.append(remaining.pop(i))
                    break

        # Append any children not in the order list
        ordered.extend(remaining)
        node.children = ordered

    # Recurse into structural children
    for child in node.children:
        if child.is_structural():
            child_path = f"{current_path}.{child.label}"
            _apply_order(child, child_path, target_order)




def verify_patch(tree1: TreeNode, tree2: TreeNode, edit_script: list) -> bool:

    print("=" * 60)
    print("  PATCH VERIFICATION")
    print("=" * 60)
    print()

    # Apply patch
    print("  Step 1: Applying edit script to source tree...")
    patched = apply_patch(tree1, edit_script)
    print()

    # Compare
    print("  Step 2: Computing TED between patched tree and target...")
    dist = tree_edit_distance(patched, tree2)
    print(f"    TED(patched, target) = {dist}")
    print()

    if dist == 0:
        print("  ✓ PATCH SUCCESSFUL: patched tree equals target tree")
        success = True
    else:
        print(f"  ✗ PATCH INCOMPLETE: residual distance = {dist}")
        print()
        print("  Patched tree:")
        print_tree(patched)
        print()
        print("  Target tree:")
        print_tree(tree2)

        # Show what still differs
        residual = compute_edit_script(patched, tree2)
        residual_edits = [op for op in residual if op["op"] != "Match"]
        if residual_edits:
            print(f"\n  Residual differences ({len(residual_edits)} operations needed):")
            for op in residual_edits:
                if op["op"] == "Update":
                    print(f"    UPDATE: {op['node1']} → {op['node2']} ({op['type']})")
                elif op["op"] == "Delete":
                    print(f"    DELETE: {op['node1']} ({op['type']})")
                elif op["op"] == "Insert":
                    print(f"    INSERT: {op['node2']} ({op['type']})")
        success = False

    print()
    return success


def save_diff(edit_script: list, source_name: str, target_name: str,
              filepath: str):
    """Save an edit script as a JSON diff file for later patching."""
    diff = {
        "source": source_name,
        "target": target_name,
        "operations": edit_script
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(diff, f, indent=2, ensure_ascii=False)
    print(f"  Diff saved to: {filepath}")


def load_diff(filepath: str) -> list:
    """Load an edit script from a JSON diff file."""
    with open(filepath, "r", encoding="utf-8") as f:
        diff = json.load(f)
    return diff.get("operations", diff.get("edit_script", []))


def tree_to_dict(node: TreeNode) -> dict:
    """Serialize a tree to a JSON-compatible dict."""
    if node.is_leaf():
        return {"label": node.label}
    return {"label": node.label,
            "children": [tree_to_dict(c) for c in node.children]}


def run_demo():
    """
    Demonstrate patching with two sample country trees.
    """
    print("=" * 60)
    print("  TREE PATCHING DEMO")
    print("=" * 60)
    print()

    # ── Build sample trees ──
    t1 = TreeNode("country", [
        TreeNode("name", [TreeNode("lebanon")]),
        TreeNode("infobox_title", [TreeNode("lebanese republic")]),
        TreeNode("capital_and_largest_city", [TreeNode("beirut")]),
        TreeNode("official_languages", [TreeNode("arabic")]),
        TreeNode("president", [TreeNode("joseph aoun")]),
        TreeNode("prime_minister", [TreeNode("nawaf salam")]),
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
        TreeNode("president", [TreeNode("bashar al-assad")]),
        TreeNode("prime_minister", [TreeNode("hussein arnous")]),
        TreeNode("area", [
            TreeNode("total", [TreeNode("185,180 km2")]),
            TreeNode("water_percent", [TreeNode("1.1")])
        ]),
        TreeNode("population", [
            TreeNode("total", [TreeNode("21,000,000")]),
            TreeNode("density", [TreeNode("113/km2")])
        ])
    ])

    print("  Source tree (Lebanon):")
    print_tree(t1)
    print()
    print("  Target tree (Syria):")
    print_tree(t2)
    print()

    # ── Compute edit script ──
    print("  Computing edit script (diff)...")
    script = compute_edit_script(t1, t2)
    print_edit_script(script)
    print()

    # ── Apply patch ──
    print("  Applying patch: Lebanon + diff → Syria")
    verify_patch(t1, t2, script)
    print()

    # ── Also test a case with structural differences ──
    print("=" * 60)
    print("  TEST: Structural insert + delete")
    print("=" * 60)
    print()

    t3 = TreeNode("country", [
        TreeNode("name", [TreeNode("lebanon")]),
        TreeNode("capital", [TreeNode("beirut")]),
        TreeNode("ethnic_groups", [TreeNode("arab")])
    ])

    t4 = TreeNode("country", [
        TreeNode("name", [TreeNode("switzerland")]),
        TreeNode("capital", [TreeNode("bern")]),
        TreeNode("gdp", [TreeNode("800 billion")])
    ])

    print("  Source (Lebanon):")
    print_tree(t3)
    print()
    print("  Target (Switzerland):")
    print_tree(t4)
    print()

    script2 = compute_edit_script(t3, t4)
    print("  Edit script:")
    print_edit_script(script2)
    print()

    print("  Applying patch...")
    verify_patch(t3, t4, script2)


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
            print("  TREE PATCHING — Real Data")
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

            print(f"\n  Source: {n1}")
            t1 = load_tree_from_json(countries[n1])
            print(f"  Target: {n2}")
            t2 = load_tree_from_json(countries[n2])
            print()

            print(f"  Source tree ({n1}, {t1.size()} nodes):")
            print_tree(t1)
            print()
            print(f"  Target tree ({n2}, {t2.size()} nodes):")
            print_tree(t2)
            print()

            # Similarity first
            sim = tree_similarity(t1, t2)
            print(f"  Similarity: structural={sim['structural_similarity_percent']}%, "
                  f"content={sim['content_similarity_percent']}%, "
                  f"overall={sim['overall_similarity_percent']}%")
            print()

            # Compute diff
            print("  Computing edit script...")
            script = compute_edit_script(t1, t2)
            print_edit_script(script)
            print()

            # Apply patch and verify
            verify_patch(t1, t2, script)

            # Save diff
            diff_path = os.path.join(
                TREE_DIR,
                f"_patch_{n1.replace(' ', '_')}_to_{n2.replace(' ', '_')}.json"
            )
            save_diff(script, n1, n2, diff_path)
            print()

            return

    # Fallback to demo
    run_demo()


if __name__ == "__main__":
    main()
