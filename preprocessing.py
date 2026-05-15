import json
import re
import os


# ─────────────────────────────────────────────
# Tree Node
# ─────────────────────────────────────────────

class TreeNode:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children else []

    def add_child(self, node):
        self.children.append(node)

    def is_leaf(self):
        return len(self.children) == 0


# ─────────────────────────────────────────────
# Normalize structural labels
# ─────────────────────────────────────────────

def normalize_label(label: str) -> str:
    label = label.lower().strip()
    label = label.replace(" ", "_").replace("-", "_")
    label = re.sub(r"[^\w]", "_", label)
    label = re.sub(r"_+", "_", label)
    return label.strip("_")


# ─────────────────────────────────────────────
# Clean scraped text
# ─────────────────────────────────────────────

def clean_content_value(value: str):

    if not value:
        return ""

    # Remove coordinate blocks
    value = re.sub(r"\d+°[\d′″\'\"\s\.NSEW°/,;]*$", "", value)

    # Remove coordinate tokens
    value = re.sub(r"\d+[°]\d*[′\']?\d*[″\"]?\s*[NSEW]", "", value)

    # Remove decimal coordinates with compass
    value = re.sub(r"\d+\.\d+°?\s*[NSEW]", "", value)

    # Remove slash coordinate pairs
    value = re.sub(r"\d+\.\d+\s*/\s*\d+\.\d+", "", value)

    # Remove citation markers
    value = re.sub(r"\[\d+\]", "", value)
    value = re.sub(r"\[[a-zA-Z]+\]", "", value)

    # Remove parentheses
    value = re.sub(r"\([^)]*\)", "", value)

    # Remove unicode invisible characters
    value = re.sub(r"[\u200b\u200c\u200d\u00a0\ufeff]", " ", value)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip().lower()

    return value


# ─────────────────────────────────────────────
# Split percentage lists (religion etc.)
# ─────────────────────────────────────────────

def split_percentages(text):

    pattern = r"\d+(?:\.\d+)?%\s*[a-zA-Z\s]+?(?=\s*\d+(?:\.\d+)?%|$)"
    matches = re.findall(pattern, text)

    result = []
    for m in matches:
        result.append(m.strip().lower())

    return result


# ─────────────────────────────────────────────
# Add leaf nodes with deduplication
# ─────────────────────────────────────────────

def _add_value_leaves(parent, value):

    cleaned = clean_content_value(value)

    if not cleaned:
        return

    percent_items = split_percentages(cleaned)

    existing = {c.label for c in parent.children}

    # Handle religion / ethnic percentages
    if percent_items:

        for item in percent_items:
            item = item.strip().lower()

            if item not in existing:
                parent.add_child(TreeNode(item))
                existing.add(item)

        return

    # Normal values
    if cleaned not in existing:
        parent.add_child(TreeNode(cleaned))


# ─────────────────────────────────────────────
# Convert JSON → Tree
# ─────────────────────────────────────────────

def convert_json_to_tree(doc):

    root = TreeNode("country")

    country = doc.get("country", "")
    if country:
        n = TreeNode("name")
        _add_value_leaves(n, country)
        root.add_child(n)

    infobox_title = doc.get("infobox_title", "")
    if infobox_title:
        n = TreeNode("infobox_title")
        _add_value_leaves(n, infobox_title)
        root.add_child(n)

    for field in doc.get("data", []):

        label = field.get("label", "")
        value = field.get("value", "")

        if not label:
            continue

        node = TreeNode(normalize_label(label))

        sub_items = field.get("sub_items", [])

        if sub_items:
            for item in sub_items:
                _add_value_leaves(node, item)

        elif value:
            _add_value_leaves(node, value)

        if node.children:
            root.add_child(node)

    for section in doc.get("sections", []):

        header = section.get("header", "")

        if not header:
            continue

        section_node = TreeNode(normalize_label(header))

        for row in section.get("rows", []):

            label = row.get("label", "")
            value = row.get("value", "")

            if not label:
                continue

            row_node = TreeNode(normalize_label(label))

            sub_items = row.get("sub_items", [])

            if sub_items:
                for item in sub_items:
                    _add_value_leaves(row_node, item)

            elif value:
                _add_value_leaves(row_node, value)

            if row_node.children:
                section_node.add_child(row_node)

        if section_node.children:
            root.add_child(section_node)

    return root


# ─────────────────────────────────────────────
# Tree → JSON
# ─────────────────────────────────────────────

def tree_to_dict(node):

    if node.is_leaf():
        return {"label": node.label}

    return {
        "label": node.label,
        "children": [tree_to_dict(c) for c in node.children]
    }


# ─────────────────────────────────────────────
# Batch preprocessing
# ─────────────────────────────────────────────

def preprocess_all(input_dir="infobox_data", output_dir="preprocessed_trees"):

    os.makedirs(output_dir, exist_ok=True)

    trees = {}

    for filename in os.listdir(input_dir):

        if not filename.endswith(".json"):
            continue

        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            doc = json.load(f)

        tree = convert_json_to_tree(doc)

        trees[filename] = tree

        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(tree_to_dict(tree), f, indent=2, ensure_ascii=False)

    print(f"[INFO] Preprocessed {len(trees)} trees")

    return trees


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    if os.path.isdir("infobox_data"):
        preprocess_all()
    else:
        print("Run extract_infoboxes.py first")
