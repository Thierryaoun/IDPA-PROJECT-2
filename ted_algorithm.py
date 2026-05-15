import json
import os
import sys
from difflib import SequenceMatcher

INF_COST = 999999.0  # Forbids cross-field content rename


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TreeNode
# ═══════════════════════════════════════════════════════════════════════════════

class TreeNode:
    def __init__(self, label: str, children: list = None):
        self.label = label
        self.children = children if children is not None else []

    def add_child(self, child):
        self.children.append(child)
        return child

    def is_leaf(self):
        return len(self.children) == 0

    def is_structural(self):
        return not self.is_leaf()

    def is_content(self):
        return self.is_leaf()

    def node_type(self):
        return "structural" if self.is_structural() else "content"

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def depth(self):
        if self.is_leaf():
            return 0
        return 1 + max(c.depth() for c in self.children)

    def leaves(self):
        if self.is_leaf():
            return [self]
        r = []
        for c in self.children:
            r.extend(c.leaves())
        return r

    def __repr__(self):
        return f"TreeNode('{self.label}', {self.node_type()}, ch={len(self.children)})"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_tree_from_dict(d):
    node = TreeNode(d["label"])
    for cd in d.get("children", []):
        node.add_child(load_tree_from_dict(cd))
    return node


def load_tree_from_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return load_tree_from_dict(json.load(f))


def structure_only_tree(node):
    """Copy keeping only structural nodes (strips all content leaves)."""
    if node.is_leaf():
        return None
    n = TreeNode(node.label)
    for c in node.children:
        if c.is_structural():
            cc = structure_only_tree(c)
            if cc:
                n.add_child(cc)
    return n


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Structural path computation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_structural_paths(root):
    
    paths = {}

    def walk(node, parent_path):
        if node.is_structural():
            my_path = f"{parent_path}.{node.label}" if parent_path else node.label
        else:
            my_path = parent_path
        paths[id(node)] = my_path
        for c in node.children:
            walk(c, my_path if node.is_structural() else parent_path)

    walk(root, "")
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Cost model with structural path enforcement
# ═══════════════════════════════════════════════════════════════════════════════

def insert_cost(node):
    return 1.0


def delete_cost(node):
    return 1.0


def rename_cost(n1, n2, paths1, paths2):

    if n1.label == n2.label:
        return 0.0
    if n1.is_structural() and n2.is_structural():
        return 1.0
    if n1.is_content() and n2.is_content():
        p1 = paths1.get(id(n1), "")
        p2 = paths2.get(id(n2), "")
        if p1 != p2:
            return INF_COST  # FORBIDDEN: different structural context
        return round(1.0 - SequenceMatcher(None, n1.label, n2.label).ratio(), 4)
    return INF_COST  # type mismatch


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Tree indexing
# ═══════════════════════════════════════════════════════════════════════════════

def _index_tree(root):
    nodes = [None]  # 1-indexed
    leftmost = [0]
    children_map = {}
    parent_map = {}
    lm = {}
    idx = [0]

    def post(node):
        ci = []
        for c in node.children:
            post(c)
            ci.append(idx[0])
        idx[0] += 1
        i = idx[0]
        nodes.append(node)
        children_map[i] = ci
        for c in ci:
            parent_map[c] = i
        lm[i] = i if node.is_leaf() else lm[ci[0]]
        leftmost.append(lm[i])

    post(root)
    n = len(nodes) - 1
    parent_map[n] = 0

    pa = [0] * (n + 1)
    for i in range(1, n + 1):
        for c in children_map[i]:
            pa[c] = i
    kr = sorted(i for i in range(1, n + 1)
                if pa[i] == 0 or leftmost[i] != leftmost[pa[i]])
    return nodes, leftmost, kr, parent_map


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Tree Edit Distance
# ═══════════════════════════════════════════════════════════════════════════════

def tree_edit_distance(t1, t2):
    """Compute Nierman-Jagadish TED with structural path enforcement."""
    p1 = _compute_structural_paths(t1)
    p2 = _compute_structural_paths(t2)
    n1, lm1, kr1, _ = _index_tree(t1)
    n2, lm2, kr2, _ = _index_tree(t2)
    sz1, sz2 = len(n1) - 1, len(n2) - 1
    TD = [[0.0] * (sz2 + 1) for _ in range(sz1 + 1)]
    for ki in kr1:
        for kj in kr2:
            _fd(ki, kj, n1, n2, lm1, lm2, TD, p1, p2)
    return TD[sz1][sz2]


def _fd(i, j, n1, n2, lm1, lm2, TD, p1, p2):
    p, q = lm1[i], lm2[j]
    m, n = i - p + 2, j - q + 2
    FD = [[0.0] * n for _ in range(m)]
    for s in range(1, m):
        FD[s][0] = FD[s - 1][0] + delete_cost(n1[s + p - 1])
    for t in range(1, n):
        FD[0][t] = FD[0][t - 1] + insert_cost(n2[t + q - 1])
    for s in range(1, m):
        for t in range(1, n):
            si, ti = s + p - 1, t + q - 1
            cd = FD[s - 1][t] + delete_cost(n1[si])
            ci = FD[s][t - 1] + insert_cost(n2[ti])
            if lm1[si] == p and lm2[ti] == q:
                cr = FD[s - 1][t - 1] + rename_cost(n1[si], n2[ti], p1, p2)
                FD[s][t] = min(cd, ci, cr)
                TD[si][ti] = FD[s][t]
            else:
                cs = FD[lm1[si] - p][lm2[ti] - q] + TD[si][ti]
                FD[s][t] = min(cd, ci, cs)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Similarity (structural + content + overall)
# ═══════════════════════════════════════════════════════════════════════════════

def tree_similarity(t1, t2):
    s1, s2 = t1.size(), t2.size()

    # Overall
    d = tree_edit_distance(t1, t2)
    total = s1 + s2
    overall = 1.0 - d / total if total > 0 else 1.0

    # Structural only
    t1s, t2s = structure_only_tree(t1), structure_only_tree(t2)
    if t1s and t2s:
        ds = tree_edit_distance(t1s, t2s)
        ts = t1s.size() + t2s.size()
        struct = 1.0 - ds / ts if ts > 0 else 1.0
    else:
        ds = max(s1, s2)
        struct = 0.0

    # Content (values of matching fields only)
    content = _content_similarity(t1, t2)

    return {
        "distance": d,
        "tree1_size": s1, "tree2_size": s2,
        "structural_similarity": round(struct, 4),
        "structural_similarity_percent": round(struct * 100, 2),
        "content_similarity": round(content, 4),
        "content_similarity_percent": round(content * 100, 2),
        "overall_similarity": round(overall, 4),
        "overall_similarity_percent": round(overall * 100, 2),
    }


def _content_similarity(t1, t2):
    """Compare leaf values only for structurally matching fields."""
    f1, f2 = _field_values(t1), _field_values(t2)
    common = set(f1) & set(f2)
    if not common:
        return 0.0
    sims = []
    for k in common:
        v1, v2 = " ".join(f1[k]), " ".join(f2[k])
        sims.append(SequenceMatcher(None, v1, v2).ratio())
    return sum(sims) / len(sims)


def _field_values(node, prefix=""):
    fields = {}
    for c in node.children:
        if c.is_structural():
            k = f"{prefix}.{c.label}" if prefix else c.label
            lv = [l.label for l in c.leaves()]
            if lv:
                fields[k] = lv
            fields.update(_field_values(c, k))
    return fields


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Edit Script (diff) Recovery
# ═══════════════════════════════════════════════════════════════════════════════

OP_MATCH = "Match"
OP_INSERT = "Insert"
OP_DELETE = "Delete"
OP_UPDATE = "Update"


def compute_edit_script(t1, t2):
    """Compute the edit script (diff) ES(T1,T2) with structural enforcement."""
    p1 = _compute_structural_paths(t1)
    p2 = _compute_structural_paths(t2)
    n1, lm1, kr1, pm1 = _index_tree(t1)
    n2, lm2, kr2, pm2 = _index_tree(t2)
    sz1, sz2 = len(n1) - 1, len(n2) - 1
    TD = [[0.0] * (sz2 + 1) for _ in range(sz1 + 1)]
    TO = [[None] * (sz2 + 1) for _ in range(sz1 + 1)]
    for ki in kr1:
        for kj in kr2:
            _fd_ops(ki, kj, n1, n2, lm1, lm2, TD, TO, p1, p2)
    ops = []
    _bt(sz1, sz2, n1, n2, lm1, lm2, TD, TO, p1, p2, ops)
    return _cleanup(ops)


def _fd_ops(i, j, n1, n2, lm1, lm2, TD, TO, p1, p2):
    p, q = lm1[i], lm2[j]
    m, n = i - p + 2, j - q + 2
    FD = [[0.0] * n for _ in range(m)]
    FO = [[None] * n for _ in range(m)]
    for s in range(1, m):
        si = s + p - 1
        FD[s][0] = FD[s - 1][0] + delete_cost(n1[si])
        FO[s][0] = ("delete", si, None)
    for t in range(1, n):
        ti = t + q - 1
        FD[0][t] = FD[0][t - 1] + insert_cost(n2[ti])
        FO[0][t] = ("insert", None, ti)
    for s in range(1, m):
        for t in range(1, n):
            si, ti = s + p - 1, t + q - 1
            cd = FD[s - 1][t] + delete_cost(n1[si])
            ci = FD[s][t - 1] + insert_cost(n2[ti])
            if lm1[si] == p and lm2[ti] == q:
                rc = rename_cost(n1[si], n2[ti], p1, p2)
                cr = FD[s - 1][t - 1] + rc
                best = min(cd, ci, cr)
                FD[s][t] = best
                # Prefer: match > delete/insert > update
                if best == cr and rc == 0:
                    FO[s][t] = ("match", si, ti)
                elif best == cd:
                    FO[s][t] = ("delete", si, None)
                elif best == ci:
                    FO[s][t] = ("insert", None, ti)
                elif best == cr and rc < INF_COST:
                    FO[s][t] = ("update", si, ti)
                else:
                    FO[s][t] = ("delete", si, None)
                TD[si][ti] = FD[s][t]
                TO[si][ti] = FO[s][t]
            else:
                cs = FD[lm1[si] - p][lm2[ti] - q] + TD[si][ti]
                best = min(cd, ci, cs)
                FD[s][t] = best
                if best == cd:
                    FO[s][t] = ("delete", si, None)
                elif best == ci:
                    FO[s][t] = ("insert", None, ti)
                else:
                    FO[s][t] = ("subtree", si, ti)


def _bt(i, j, n1, n2, lm1, lm2, TD, TO, p1, p2, ops):
    if i == 0 and j == 0:
        return
    if i == 0:
        for t in range(1, j + 1):
            ops.append({"op": OP_INSERT, "node2": n2[t].label,
                        "type": n2[t].node_type(),
                        "path": p2.get(id(n2[t]), "")})
        return
    if j == 0:
        for s in range(1, i + 1):
            ops.append({"op": OP_DELETE, "node1": n1[s].label,
                        "type": n1[s].node_type(),
                        "path": p1.get(id(n1[s]), "")})
        return
    info = TO[i][j]
    if not info:
        return
    op, si, ti = info
    if op == "match":
        _bt(i - 1, j - 1, n1, n2, lm1, lm2, TD, TO, p1, p2, ops)
        ops.append({"op": OP_MATCH, "node1": n1[si].label, "node2": n2[ti].label,
                     "type": n1[si].node_type(), "path": p1.get(id(n1[si]), "")})
    elif op == "update":
        _bt(i - 1, j - 1, n1, n2, lm1, lm2, TD, TO, p1, p2, ops)
        ut = "structural" if n1[si].is_structural() and n2[ti].is_structural() else \
             "content" if n1[si].is_content() and n2[ti].is_content() else "mixed"
        ops.append({"op": OP_UPDATE, "node1": n1[si].label, "node2": n2[ti].label,
                     "type": ut, "path": p1.get(id(n1[si]), "")})
    elif op == "delete":
        _bt(i - 1, j, n1, n2, lm1, lm2, TD, TO, p1, p2, ops)
        ops.append({"op": OP_DELETE, "node1": n1[si].label,
                     "type": n1[si].node_type(), "path": p1.get(id(n1[si]), "")})
    elif op == "insert":
        _bt(i, j - 1, n1, n2, lm1, lm2, TD, TO, p1, p2, ops)
        ops.append({"op": OP_INSERT, "node2": n2[ti].label,
                     "type": n2[ti].node_type(), "path": p2.get(id(n2[ti]), "")})
    elif op == "subtree":
        _bt(lm1[si] - 1, lm2[ti] - 1, n1, n2, lm1, lm2, TD, TO, p1, p2, ops)
        _bt(si, ti, n1, n2, lm1, lm2, TD, TO, p1, p2, ops)


def _cleanup(ops):
    """
    Post-process: when DP produces UPDATE(X→Y) + INSERT(X) in same field,
    rewrite as MATCH(X) + INSERT(Y). Same cost, much cleaner to read.
    """
    ins = {}
    for idx, op in enumerate(ops):
        if op["op"] == OP_INSERT and op["type"] == "content":
            ins[(op.get("path", ""), op["node2"])] = idx
    fixes = []
    for idx, op in enumerate(ops):
        if op["op"] == OP_UPDATE and op["type"] == "content":
            key = (op.get("path", ""), op["node1"])
            if key in ins:
                fixes.append((idx, ins[key], op["node1"], op["node2"]))
    for ui, ii, src, dst in reversed(fixes):
        ops[ui] = {"op": OP_MATCH, "node1": src, "node2": src,
                    "type": "content", "path": ops[ui].get("path", "")}
        ops[ii] = {"op": OP_INSERT, "node2": dst,
                    "type": "content", "path": ops[ii].get("path", "")}
    return ops


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Display
# ═══════════════════════════════════════════════════════════════════════════════

def print_tree(node, prefix="", is_last=True, is_root=True):
    conn = "" if is_root else ("└── " if is_last else "├── ")
    tag = "S" if node.is_structural() else "C"
    print(f"{prefix}{conn}[{tag}] {node.label}")
    cp = prefix + ("" if is_root else ("    " if is_last else "│   "))
    for i, c in enumerate(node.children):
        print_tree(c, cp, i == len(node.children) - 1, False)


def print_edit_script(ops, show_matches=False):
    edits = [o for o in ops if o["op"] != OP_MATCH]
    matches = [o for o in ops if o["op"] == OP_MATCH]
    se = [o for o in edits if o["type"] == "structural"]
    ce = [o for o in edits if o["type"] == "content"]

    print(f"    Summary: {len(matches)} matches, {len(edits)} edits "
          f"({len(se)} structural, {len(ce)} content)")
    print()

    if se:
        print("    STRUCTURAL CHANGES")
        print("    " + "─" * 50)
        for o in se:
            field = o.get("path", "").split(".")[-1] or o.get("node1", "")
            if o["op"] == OP_UPDATE:
                print(f"      UPDATE field: {o['node1']}  →  {o['node2']}")
            elif o["op"] == OP_DELETE:
                print(f"      DELETE field: {o['node1']}")
            elif o["op"] == OP_INSERT:
                print(f"      INSERT field: {o['node2']}")
        print()

    if ce:
        print("    CONTENT CHANGES")
        print("    " + "─" * 50)
        for o in ce:
            field = o.get("path", "").split(".")[-1] or "?"
            if o["op"] == OP_UPDATE:
                print(f"      UPDATE {field}: {o['node1']}  →  {o['node2']}")
            elif o["op"] == OP_DELETE:
                print(f"      DELETE {field}: {o['node1']}")
            elif o["op"] == OP_INSERT:
                print(f"      INSERT {field}: {o['node2']}")
        print()

    if show_matches and matches:
        print("    MATCHES")
        print("    " + "─" * 50)
        for o in matches:
            print(f"      MATCH: {o['node1']}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Validation test
# ═══════════════════════════════════════════════════════════════════════════════

def run_validation():
    print("=" * 60)
    print("  VALIDATION: Structural enforcement")
    print("=" * 60)
    results = []

    # A: Same field → UPDATE allowed
    t1 = TreeNode("country", [TreeNode("president", [TreeNode("joseph aoun")])])
    t2 = TreeNode("country", [TreeNode("president", [TreeNode("bashar al-assad")])])
    s = compute_edit_script(t1, t2)
    cu = [o for o in s if o["op"] == OP_UPDATE and o["type"] == "content"]
    ok = any(o["node1"] == "joseph aoun" and o["node2"] == "bashar al-assad" for o in cu)
    results.append(ok)
    print(f"\n  A. Same field update: {'✓' if ok else '✗'}")

    # B: Different fields → NO cross-field update
    t1 = TreeNode("country", [
        TreeNode("president", [TreeNode("joseph aoun")]),
        TreeNode("official_languages", [TreeNode("arabic")])])
    t2 = TreeNode("country", [
        TreeNode("president", [TreeNode("bashar al-assad")]),
        TreeNode("official_languages", [TreeNode("french")])])
    s = compute_edit_script(t1, t2)
    cu = [o for o in s if o["op"] == OP_UPDATE and o["type"] == "content"]
    cross = any((o["node1"] == "joseph aoun" and o["node2"] == "french") or
                (o["node1"] == "arabic" and o["node2"] == "bashar al-assad") for o in cu)
    ok = not cross
    results.append(ok)
    print(f"  B. No cross-field: {'✓' if ok else '✗'}")
    vp = any(o["node1"] == "joseph aoun" and o["node2"] == "bashar al-assad" for o in cu)
    vl = any(o["node1"] == "arabic" and o["node2"] == "french" for o in cu)
    results.extend([vp, vl])
    print(f"  B. president update: {'✓' if vp else '✗'}")
    print(f"  B. language update:  {'✓' if vl else '✗'}")

    # C: area.total vs population.total → NO cross-path
    t1 = TreeNode("country", [TreeNode("area", [TreeNode("total", [TreeNode("10452")])])])
    t2 = TreeNode("country", [TreeNode("population", [TreeNode("total", [TreeNode("5000000")])])])
    s = compute_edit_script(t1, t2)
    cross = any(o["op"] == OP_UPDATE and o["type"] == "content" and
                o["node1"] == "10452" and o["node2"] == "5000000" for o in s)
    ok = not cross
    results.append(ok)
    print(f"  C. No cross-path:   {'✓' if ok else '✗'}")

    # D: One lang vs two → MATCH existing, INSERT new
    t1 = TreeNode("country", [TreeNode("official_languages", [TreeNode("arabic")])])
    t2 = TreeNode("country", [TreeNode("official_languages", [TreeNode("arabic"), TreeNode("french")])])
    s = compute_edit_script(t1, t2)
    am = any(o["op"] == OP_MATCH and o["node1"] == "arabic" for o in s)
    fi = any(o["op"] == OP_INSERT and o["node2"] == "french" for o in s)
    bad = any(o["op"] == OP_UPDATE and o["node1"] == "arabic" and o["node2"] == "french" for o in s)
    ok = am and fi and not bad
    results.append(ok)
    print(f"  D. Match+Insert:    {'✓' if ok else '✗'}")

    # E: Identical → TED=0
    t = TreeNode("country", [TreeNode("capital", [TreeNode("beirut")])])
    ok = tree_edit_distance(t, t) == 0
    results.append(ok)
    print(f"  E. Identical TED=0: {'✓' if ok else '✗'}")

    # F: Symmetry
    d1 = tree_edit_distance(t1, t2)
    d2 = tree_edit_distance(t2, t1)
    ok = d1 == d2
    results.append(ok)
    print(f"  F. Symmetry:        {'✓' if ok else '✗'}")

    passed = all(results)
    print(f"\n  {'★ ALL TESTS PASSED ★' if passed else '✗ SOME TESTS FAILED'}\n")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    run_validation()

    TREE_DIR = "preprocessed_trees"
    if not os.path.isdir(TREE_DIR):
        print(f"[INFO] '{TREE_DIR}/' not found. Run preprocessing.py first.")
        return

    files = sorted(f for f in os.listdir(TREE_DIR)
                   if f.endswith(".json") and not f.startswith("_"))
    if len(files) < 2:
        print(f"[ERROR] Need ≥2 files, found {len(files)}."); return

    countries = {}
    for f in files:
        countries[f.replace(".json", "").replace("_", " ")] = os.path.join(TREE_DIR, f)
    names = list(countries.keys())

    print("=" * 60)
    print("  NIERMAN-JAGADISH TED — FINAL")
    print("=" * 60)
    print(f"\n  {len(names)} countries in '{TREE_DIR}/'")
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

    print(f"\n  Tree 1: {n1} ({t1.size()} nodes)")
    print_tree(t1)
    print(f"\n  Tree 2: {n2} ({t2.size()} nodes)")
    print_tree(t2)

    # Similarity
    print("\n" + "=" * 60)
    print("  SIMILARITY")
    print("=" * 60 + "\n")
    sim = tree_similarity(t1, t2)
    print(f"    TED = {sim['distance']}")
    print(f"    Structural similarity : {sim['structural_similarity_percent']:.2f}%")
    print(f"    Content similarity    : {sim['content_similarity_percent']:.2f}%")
    print(f"    Overall similarity    : {sim['overall_similarity_percent']:.2f}%")

    # Edit script
    print("\n" + "=" * 60)
    print(f"  EDIT SCRIPT (diff): {n1} → {n2}")
    print("=" * 60 + "\n")
    script = compute_edit_script(t1, t2)
    print_edit_script(script)

    # Save diff as JSON (for patching stage)
    result = {
        "source": n1, "target": n2,
        "source_nodes": sim["tree1_size"], "target_nodes": sim["tree2_size"],
        "ted": sim["distance"],
        "structural_similarity": sim["structural_similarity_percent"],
        "content_similarity": sim["content_similarity_percent"],
        "overall_similarity": sim["overall_similarity_percent"],
        "edit_script": script,
    }
    out = os.path.join(TREE_DIR,
                       f"_diff_{n1.replace(' ', '_')}_vs_{n2.replace(' ', '_')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Diff saved to: {out}\n")


if __name__ == "__main__":
    main()
