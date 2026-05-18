import json
import html
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from xml.etree import ElementTree

INF_COST = 999999.0  # Forbids cross-field content rename


# Semantic preprocessing configuration. Keep these dictionaries in one place so
# field aliases, ignored metadata, and weights can be tuned without touching TED.
FIELD_ALIASES = {
    "official_language": "languages.official",
    "official_languages": "languages.official",
    "official_language_and_national_language": "languages.official",
    "official_and_national_language": "languages.official",
    "official_language_or_national_language": "languages.official",
    "national_language": "languages.official",
    "national_languages_and_official_languages": "languages.official",
    "recognized_national_languages": "languages.recognized",
    "recognized_languages": "languages.recognized",
    "recognised_minority_language": "languages.recognized",
    "recognised_minority_languages": "languages.recognized",
    "national_languages": "languages.recognized",
    "national_minority_languages": "languages.recognized",
    "national_minority_language": "languages.recognized",
    "regional_languages": "languages.recognized",
    "regional_language": "languages.recognized",
    "co_official_languages": "languages.recognized",
    "co_official_language": "languages.recognized",
    "minority_languages": "languages.recognized",
    "minority_language": "languages.recognized",
    "spoken_languages": "languages.spoken",
    "local_vernacular": "languages.spoken",
    "national_vernacular": "languages.spoken",
    "foreign_languages": "languages.spoken",
    "significant_language": "languages.spoken",
    "other_languages": "languages.spoken",
    "special_status": "languages.recognized",
    "ethnic_groups": "demographics.ethnic_groups",
    "ethnic_group": "demographics.ethnic_groups",
    "religion": "demographics.religion",
    "religions": "demographics.religion",
    "government_type": "politics.government",
    "government": "politics.government",
    "politics": "politics.government",
    "capital": "identity.capital",
    "capital_city": "identity.capital",
    "capital_and_largest_city": "identity.capital",
    "currency": "economy.currency",
    "currencies": "economy.currency",
    "gdp_ppp": "economy.gdp.ppp",
    "gdp_nominal": "economy.gdp.nominal",
    "population_estimate": "demographics.population",
    "population_census": "demographics.population",
    "density": "demographics.population.density",
    "hdi": "demographics.hdi",
    "gini": "demographics.gini",
    "establishment": "history.establishment",
    "establishment_history": "history.establishment",
    "formation": "history.establishment",
    "area": "geography.area",
}

IGNORE_FIELDS = {
    "name", "native_name", "conventional_long_name", "common_name",
    "infobox_title", "image_flag", "image_coat", "flag_caption",
    "symbol_type", "national_anthem", "iso3166code", "iso_3166_code",
    "calling_code", "cctld", "internet_tld", "drives_on",
    "driving_side", "date_format", "utc_offset", "time_zone",
    "summer_dst", "coordinates", "latd", "longd", "latm", "longm",
    "leader_name", "leader_title", "president", "prime_minister",
    "supreme_leader", "monarch", "king", "queen", "governor_general",
    "chief_justice", "speaker", "parliament_speaker",
    "speaker_of_the_parliament", "council_president", "assembly_president",
    "knesset_speaker", "legislature", "largest_city", "demonym",
    "identity.capital",
    # Legislature chamber names — differ between every parliamentary country → noise
    "lower_house", "upper_house",
    # Demonyms and nationality labels — not semantically meaningful for clustering
    "demonyms", "nationality",
    # Additional leader/title fields not caught by the generic list above
    "emir", "crown_prince", "crown_prince_and_prime_minister",
    "chancellor", "riksdag_speaker",
    # Data artifacts from Wikipedia infobox scraping (quarter/month keys)
    "q1", "q2", "q3", "q4",
    "january", "february", "march", "april", "june",
    "july", "august", "september", "october", "november", "december",
    # Country-name-in-regional-language fields (Spain, etc.) — these are native_name variants,
    # not linguistic content. A country name written in Basque/Catalan/Galician tells us nothing
    # about linguistic policy and inflates the structural distance for Spain unfairly.
    "basque", "catalan", "catalan_valencian", "valencian",
    "galician", "occitan_aranese", "aranese", "aragonese", "asturian",
    "breton", "scots_gaelic", "welsh", "irish", "cornish", "manx",
    # Script/writing system fields — not semantically meaningful for country clustering
    "official_script", "script",
    # Additional leader/title fields
    "emperor", "empress", "sultan", "caliph", "head_of_state", "head_of_government",
    "president_of_the_senate", "president_of_the_national_assembly",
    "president_of_the_congress_of_deputies",
}

FIELD_WEIGHTS = {
    "languages.official": 2.0,
    "languages.recognized": 1.5,
    "languages.spoken": 1.0,
    "demographics.religion": 2.0,
    "demographics.ethnic_groups": 1.5,
    "politics.government": 1.5,
    "geography.region": 2.0,
    "geography.continent": 2.0,
    "economy.currency": 0.8,
    "economy.gdp": 0.5,
    "demographics.population": 0.5,
    "history.establishment": 0.3,
    "geography.area": 0.8,
    "demographics.hdi": 0.8,
    "demographics.gini": 0.8,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TreeNode
# ═══════════════════════════════════════════════════════════════════════════════

class TreeNode:
    def __init__(self, label: str, children: list = None, weight: float = 1.0):
        self.label = label
        self.children = children if children is not None else []
        self.weight = weight

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


def parse_country_xml(filepath):
    """Parse a simple XML tree into TreeNode form.

    The current project stores extracted infoboxes as JSON, but this keeps the
    semantic pipeline usable if XML files are added later.
    """
    root_el = ElementTree.parse(filepath).getroot()

    def walk(el):
        node = TreeNode(clean_value(el.tag) or "node")
        text = clean_value(el.text or "")
        if text:
            node.add_child(TreeNode(text))
        for child in el:
            node.add_child(walk(child))
        return node

    return walk(root_el)


def clean_value(value: str) -> str:
    """Clean noisy scraped labels/values before tree comparison."""
    if value is None:
        return ""
    value = html.unescape(str(value))
    value = unicodedata.normalize("NFKC", value)
    replacements = {
        "â€“": "-", "â€”": "-", "â€²": "'", "â€³": '"',
        "Â°": "°", "Â": "", "\u00a0": " ",
    }
    for bad, good in replacements.items():
        value = value.replace(bad, good)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\[(?:\d+|note\s*\d+|[a-zA-Z])\]", " ", value, flags=re.I)
    value = re.sub(r"/\s*\d{1,3}(?:\.\d+)?\.?", " ", value)
    value = re.sub(r"\d+(?:\.\d+)?°?\s*[NSEW]\b", " ", value, flags=re.I)
    value = re.sub(r"[\u200b\u200c\u200d\ufeff]", " ", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    meaningless = {"", "-", "—", "n/a", "na", "none", "unknown", "see text"}
    return "" if value in meaningless else value


def normalize_field_name(field_name: str) -> str:
    """Map raw field labels to canonical semantic labels."""
    cleaned = clean_value(field_name)
    key = cleaned.replace(" ", "_").replace("-", "_")
    key = re.sub(r"[^\w]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    key = re.sub(r"_\d{4}(?:_est|_estimate|_census)?$", "", key)
    key = re.sub(r"_\d{4}_(?:est|estimate|census)$", "", key)
    if key.startswith("ethnic_groups"):
        return "demographics.ethnic_groups"
    if key.startswith("religion"):
        return "demographics.religion"
    if key.startswith("hdi"):
        return "demographics.hdi"
    if key.startswith("gini"):
        return "demographics.gini"
    if key.endswith("_estimate") or key.endswith("_census") or "population" in key:
        return FIELD_ALIASES.get(key, "demographics.population")
    # Population estimates embedded in weird field names (e.g. "metropolitan_france_estimate_as_of_january")
    if "_estimate_as_of_" in key or (re.search(r"_as_of_\d{4}$", key)):
        return "demographics.population"
    # Pure month-name keys (population date artifacts: "january", "february_1", etc.)
    if re.match(r"^(january|february|march|april|june|july|august|september|october|november|december)(_\d+)?$", key):
        return "demographics.population"
    return FIELD_ALIASES.get(key, key)


def should_ignore_field(field_path: str, value: str = "") -> bool:
    parts = [p for p in field_path.lower().split(".") if p]
    leaf = parts[-1] if parts else field_path.lower()
    if field_path.lower() in IGNORE_FIELDS or leaf in IGNORE_FIELDS:
        return True
    if leaf == "capital" and re.search(r"/\s*\d", value or ""):
        return True
    # Any field naming a specific government officer: "president_of_the_X", "chief_X_of_the_Y",
    # "speaker_of_the_X", etc.  These are leader names in disguise.
    _OFFICER_PREFIXES = (
        "president_of_the_", "speaker_of_the_", "chair_of_the_",
        "chief_justice_of_the_", "secretary_of_the_", "minister_of_the_",
        "director_of_the_", "head_of_the_", "commissioner_of_the_",
    )
    if any(leaf.startswith(p) for p in _OFFICER_PREFIXES):
        return True
    if leaf.endswith("_of_the_parliament") or leaf.endswith("_of_parliament"):
        return True
    # Population artifact: month name followed by digit (e.g. "january_1", "october_1")
    _MONTHS = {
        "january", "february", "march", "april", "june",
        "july", "august", "september", "october", "november", "december",
    }
    if re.match(r"^(" + "|".join(_MONTHS) + r")_\d", leaf):
        return True
    return False


def get_field_weight(field_path: str) -> float:
    path = field_path.lower()
    best_len, best_weight = -1, 1.0
    for prefix, weight in FIELD_WEIGHTS.items():
        if path == prefix or path.startswith(prefix + "."):
            if len(prefix) > best_len:
                best_len, best_weight = len(prefix), weight
    return best_weight


def _field_matches_filter(canonical: str, field_filter: set) -> bool:
    """Return True if *canonical* starts with any prefix in *field_filter*."""
    for prefix in field_filter:
        if canonical == prefix or canonical.startswith(prefix + "."):
            return True
    return False


def normalize_country_tree(
    tree: TreeNode,
    debug: dict = None,
    field_filter: set | None = None,
) -> TreeNode:
    """Build a weighted semantic tree from a raw/preprocessed infobox tree.

    Parameters
    ----------
    field_filter : optional set of canonical field prefixes to include.
        If None, all fields are included (default behaviour).
        Example: {"economy.gdp", "geography.area"} compares only on
        GDP and land area.
    """
    debug = debug if debug is not None else {}
    debug.setdefault("fields_removed", [])
    debug.setdefault("fields_normalized", [])
    root = TreeNode("country", weight=1.0)
    raw_fields = _collect_raw_fields(tree)

    for raw_path, values in raw_fields:
        canonical = _canonical_path(raw_path)
        cleaned_values = [clean_value(v) for v in values]
        cleaned_values = [v for v in cleaned_values if v]
        if not cleaned_values:
            continue
        if should_ignore_field(canonical, " ".join(cleaned_values)):
            debug["fields_removed"].append({"field": raw_path, "canonical": canonical})
            continue
        # Field filter: skip fields not in the selected set
        if field_filter is not None and not _field_matches_filter(canonical, field_filter):
            continue
        if canonical != raw_path:
            debug["fields_normalized"].append({"from": raw_path, "to": canonical})
        _add_semantic_field(root, canonical, cleaned_values)

    _merge_duplicate_structures(root)
    _cap_direct_value_weights(root)
    _sort_tree(root)
    return root


def _collect_raw_fields(node, prefix=""):
    fields = []
    for child in node.children:
        if child.is_structural():
            path = f"{prefix}.{child.label}" if prefix else child.label
            direct = [c.label for c in child.children if c.is_leaf()]
            if direct:
                fields.append((path, direct))
            fields.extend(_collect_raw_fields(child, path))
    return fields


def _canonical_path(raw_path):
    parts = [p for p in raw_path.split(".") if p and p != "country"]
    if not parts:
        return "country"
    # Repair GDP rows that were scraped under population with generic siblings.
    if parts[0] == "population":
        leaf = parts[-1]
        if leaf in {"gdp_ppp", "gdp_nominal"}:
            return normalize_field_name(leaf)
        if leaf in {"total", "per_capita"}:
            # The original flat tree cannot always tell PPP vs nominal. Keep it
            # under economy.gdp so it is not treated as population.
            return f"economy.gdp.{leaf}"
        return normalize_field_name(leaf)
    if len(parts) == 1:
        return normalize_field_name(parts[0])
    head = normalize_field_name(parts[0])
    leaf = normalize_field_name(parts[-1])
    if head == "history.establishment":
        return f"{head}.{leaf}"
    if head in {"geography.area", "economy.gdp"}:
        return f"{head}.{leaf}"
    return leaf if "." in leaf else f"{head}.{leaf}"


def _add_semantic_field(root, canonical_path, values):
    current = root
    parts = canonical_path.split(".")
    for i, part in enumerate(parts):
        path = ".".join(parts[:i + 1])
        child = _find_child(current, part)
        if not child:
            child = current.add_child(TreeNode(part, weight=get_field_weight(path)))
        current = child
    if (_is_distribution_field(canonical_path) or _is_set_field(canonical_path)) and len(values) > 1:
        values = [" | ".join(sorted(values))]
    existing = {c.label for c in current.children if c.is_leaf()}
    for value in values:
        if value not in existing:
            current.add_child(TreeNode(value, weight=get_field_weight(canonical_path)))
            existing.add(value)


def _find_child(node, label):
    for child in node.children:
        if child.label == label and child.is_structural():
            return child
    return None


def _merge_duplicate_structures(node):
    merged = {}
    new_children = []
    for child in node.children:
        if child.is_structural():
            _merge_duplicate_structures(child)
            if child.label in merged:
                merged[child.label].children.extend(child.children)
            else:
                merged[child.label] = child
                new_children.append(child)
        elif child.label not in {c.label for c in new_children if c.is_leaf()}:
            new_children.append(child)
    node.children = new_children


def _cap_direct_value_weights(node):
    """Keep repeated values from dominating a semantic field's total cost."""
    direct_leaves = [c for c in node.children if c.is_leaf()]
    if direct_leaves:
        per_value = getattr(node, "weight", 1.0) / len(direct_leaves)
        for leaf in direct_leaves:
            leaf.weight = per_value
    for child in node.children:
        if child.is_structural():
            _cap_direct_value_weights(child)


def _sort_tree(node):
    node.children.sort(key=lambda c: (0 if c.is_structural() else 1, c.label))
    for child in node.children:
        if child.is_structural():
            _sort_tree(child)


def weighted_tree_cost(root):
    return sum(_subtree_delete_cost(c) for c in root.children) + root.weight


def _subtree_delete_cost(node):
    return node.weight + sum(_subtree_delete_cost(c) for c in node.children)


def cost_del_tree(node):
    """CostDelTree: weighted cost to delete the entire subtree rooted at node."""
    return delete_cost(node) + sum(cost_del_tree(c) for c in node.children)


def cost_ins_tree(node):
    """CostInsTree: weighted cost to insert the entire subtree rooted at node."""
    return insert_cost(node) + sum(cost_ins_tree(c) for c in node.children)


def _unit_subtree_size(node):
    """Unit cost to delete/insert an entire subtree (one per node)."""
    return 1 + sum(_unit_subtree_size(c) for c in node.children)


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
    return getattr(node, "weight", 1.0)


def delete_cost(node):
    return getattr(node, "weight", 1.0)


def rename_cost(n1, n2, paths1, paths2):
    l1, l2 = clean_value(n1.label), clean_value(n2.label)
    weight = (getattr(n1, "weight", 1.0) + getattr(n2, "weight", 1.0)) / 2
    if l1 == l2:
        return 0.0
    if n1.is_structural() and n2.is_structural():
        c1, c2 = normalize_field_name(l1), normalize_field_name(l2)
        if c1 == c2:
            return 0.0
        if _related_fields(c1, c2):
            return round(0.2 * weight, 4)
        return weight
    if n1.is_content() and n2.is_content():
        p1 = paths1.get(id(n1), "")
        p2 = paths2.get(id(n2), "")
        canonical_path = _canonical_path(p1)
        if canonical_path != _canonical_path(p2):
            return INF_COST  # FORBIDDEN: different structural context
        if l1 == l2:
            return 0.0
        return round(_value_rename_cost(canonical_path, l1, l2, weight), 4)
    return INF_COST  # type mismatch


def _related_fields(c1, c2):
    p1, p2 = c1.split("."), c2.split(".")
    if p1[0] == p2[0] and p1[0] in {
        "languages", "demographics", "economy", "politics", "history", "geography"
    }:
        return True
    return SequenceMatcher(None, c1, c2).ratio() >= 0.65


def _value_rename_cost(path, value1, value2, weight):
    if _is_date_field(path):
        return (1.0 - _date_similarity([value1], [value2])) * weight
    if _is_enum_field(path):
        return (1.0 - _enum_similarity([value1], [value2])) * weight
    if _is_distribution_field(path):
        return _distribution_value_cost(value1, value2, weight)
    if _is_numeric_field(path):
        n1, n2 = _extract_number(value1), _extract_number(value2)
        if n1 is not None and n2 is not None and n1 > 0 and n2 > 0:
            return (1.0 - min(n1, n2) / max(n1, n2)) * weight
    if _is_set_field(path):
        sim = _set_similarity([value1], [value2])
        if sim > 0:
            return (1.0 - sim) * weight
    if "government" in path:
        return (1.0 - _government_similarity([value1], [value2])) * weight
    ratio = SequenceMatcher(None, value1, value2).ratio()
    return (1.0 - ratio) * weight


def _distribution_value_cost(value1, value2, weight):
    d1, d2 = _parse_distribution([value1]), _parse_distribution([value2])
    if d1 and d2:
        keys = set(d1) | set(d2)
        total = sum(max(d1.get(k, 0.0), d2.get(k, 0.0)) for k in keys)
        if total > 0:
            overlap = sum(min(d1.get(k, 0.0), d2.get(k, 0.0)) for k in keys)
            return (1.0 - overlap / total) * weight
    pct1, cat1 = _split_percent_category(value1)
    pct2, cat2 = _split_percent_category(value2)
    if not cat1 or not cat2:
        return (1.0 - SequenceMatcher(None, value1, value2).ratio()) * weight
    category_overlap = _set_similarity([cat1], [cat2])
    if category_overlap == 0:
        category_overlap = SequenceMatcher(None, cat1, cat2).ratio()
    percent_similarity = 1.0
    if pct1 is not None and pct2 is not None:
        percent_similarity = 1.0 - min(abs(pct1 - pct2) / 100.0, 1.0)
    similarity = 0.75 * category_overlap + 0.25 * percent_similarity
    return (1.0 - similarity) * weight


def _split_percent_category(value):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", value)
    pct = float(m.group(1)) if m else None
    category = re.sub(r"\d+(?:\.\d+)?\s*%", " ", value)
    category = re.sub(r"[^a-z0-9]+", " ", category).strip()
    return pct, category


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
# 6. Tree Edit Distance — Nierman-Jagadish recursive algorithm
#
# Algorithm (from the literature):
#   Input:  A, B — document trees
#   Output: TED(A, B)
#
#   M = Degree(A),  N = Degree(B)          // first-level subtree counts
#   Dist[0..M][0..N]
#   Dist[0][0] = CostUpd(R(A), R(B))       // root update cost
#   For i = 1..M: Dist[i][0] = Dist[i-1][0] + CostDelTree(Ai)
#   For j = 1..N: Dist[0][j] = Dist[0][j-1] + CostInsTree(Bj)
#   For i = 1..M, j = 1..N:
#     Dist[i][j] = min{
#       Dist[i-1][j-1] + TED(Ai, Bj),     // recursive call
#       Dist[i-1][j]   + CostDelTree(Ai),
#       Dist[i][j-1]   + CostInsTree(Bj)
#     }
#   Return Dist[M][N]
# ═══════════════════════════════════════════════════════════════════════════════

def tree_edit_distance(A, B, _paths1=None, _paths2=None, _memo=None):
    """Nierman-Jagadish TED: recursive, top-down, weighted cost model."""
    if _paths1 is None:
        _paths1 = _compute_structural_paths(A)
    if _paths2 is None:
        _paths2 = _compute_structural_paths(B)
    if _memo is None:
        _memo = {}

    key = (id(A), id(B))
    if key in _memo:
        return _memo[key]

    M = len(A.children)   # Degree(A): number of first-level subtrees
    N = len(B.children)   # Degree(B): number of first-level subtrees

    # Dist[0..M][0..N]
    Dist = [[0.0] * (N + 1) for _ in range(M + 1)]

    # Line 4: Dist[0][0] = CostUpd(R(A), R(B))
    Dist[0][0] = rename_cost(A, B, _paths1, _paths2)

    # Line 5: Dist[i][0] = Dist[i-1][0] + CostDelTree(Ai)
    for i in range(1, M + 1):
        Dist[i][0] = Dist[i - 1][0] + cost_del_tree(A.children[i - 1])

    # Line 6: Dist[0][j] = Dist[0][j-1] + CostInsTree(Bj)
    for j in range(1, N + 1):
        Dist[0][j] = Dist[0][j - 1] + cost_ins_tree(B.children[j - 1])

    # Lines 7-17: main recurrence with recursive TED(Ai, Bj)
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            Ai = A.children[i - 1]
            Bj = B.children[j - 1]
            Dist[i][j] = min(
                Dist[i - 1][j - 1] + tree_edit_distance(Ai, Bj, _paths1, _paths2, _memo),
                Dist[i - 1][j]     + cost_del_tree(Ai),
                Dist[i][j - 1]     + cost_ins_tree(Bj),
            )

    result = Dist[M][N]
    _memo[key] = result
    return result


def tree_edit_distance_legacy(A, B, _paths1=None, _paths2=None, _memo=None):
    """Nierman-Jagadish TED: recursive, top-down, unweighted (unit) cost model."""
    if _paths1 is None:
        _paths1 = _compute_structural_paths(A)
    if _paths2 is None:
        _paths2 = _compute_structural_paths(B)
    if _memo is None:
        _memo = {}

    key = (id(A), id(B))
    if key in _memo:
        return _memo[key]

    M = len(A.children)
    N = len(B.children)

    Dist = [[0.0] * (N + 1) for _ in range(M + 1)]

    # Dist[0][0] = CostUpd(R(A), R(B)) with unit/string-similarity cost
    Dist[0][0] = _legacy_rename_cost(A, B, _paths1, _paths2)

    # Boundary: cost to delete/insert entire subtrees (unit cost per node)
    for i in range(1, M + 1):
        Dist[i][0] = Dist[i - 1][0] + _unit_subtree_size(A.children[i - 1])

    for j in range(1, N + 1):
        Dist[0][j] = Dist[0][j - 1] + _unit_subtree_size(B.children[j - 1])

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            Ai = A.children[i - 1]
            Bj = B.children[j - 1]
            Dist[i][j] = min(
                Dist[i - 1][j - 1] + tree_edit_distance_legacy(Ai, Bj, _paths1, _paths2, _memo),
                Dist[i - 1][j]     + _unit_subtree_size(Ai),
                Dist[i][j - 1]     + _unit_subtree_size(Bj),
            )

    result = Dist[M][N]
    _memo[key] = result
    return result


def _legacy_rename_cost(n1, n2, paths1, paths2):
    if n1.label == n2.label:
        return 0.0
    if n1.is_structural() and n2.is_structural():
        return 1.0
    if n1.is_content() and n2.is_content():
        p1 = paths1.get(id(n1), "")
        p2 = paths2.get(id(n2), "")
        if p1 != p2:
            return INF_COST
        return round(1.0 - SequenceMatcher(None, n1.label, n2.label).ratio(), 4)
    return INF_COST


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


def compute_similarity(t1, t2, use_semantic_preprocessing=True,
                       save_debug_path=None, top_edits=25):
    """Compare two countries with raw TED plus optional semantic TED.

    Nierman-Jagadish TED remains the core distance algorithm. Semantic mode only
    changes the tree representation and weighted cost model before TED runs.
    """
    raw_distance = tree_edit_distance_legacy(t1, t2)
    raw_total = t1.size() + t2.size()
    raw_similarity = 1.0 - raw_distance / raw_total if raw_total else 1.0
    result = {
        "raw_ted_distance": raw_distance,
        "raw_tree_similarity": round(raw_similarity, 4),
        "raw_tree_similarity_percent": round(raw_similarity * 100, 2),
        "raw_tree1_size": t1.size(),
        "raw_tree2_size": t2.size(),
        "warning": (
            "This score measures normalized infobox tree similarity, "
            "not real-world geopolitical similarity."
        ),
    }
    if not use_semantic_preprocessing:
        result["semantic_preprocessing"] = False
        return result

    debug1, debug2 = {}, {}
    nt1, nt2 = normalize_country_tree(t1, debug1), normalize_country_tree(t2, debug2)
    distance = tree_edit_distance(nt1, nt2)
    max_cost = weighted_tree_cost(nt1) + weighted_tree_cost(nt2)
    weighted_similarity = 1.0 - distance / max_cost if max_cost else 1.0
    script = compute_edit_script(nt1, nt2)
    edits = [op for op in script if op["op"] != OP_MATCH]

    # Annotate each edit operation with its semantic classification
    classified_edits = [{**op, "classification": classify_edit_op(op)} for op in edits]
    expected_domain  = [e for e in classified_edits if e["classification"] == "expected_domain_difference"]
    structural_diffs = [e for e in classified_edits if e["classification"] == "structural_difference"]
    content_updates  = [e for e in classified_edits if e["classification"] == "content_update"]

    debug = {
        "tree1": debug1,
        "tree2": debug2,
        "top_edit_operations": classified_edits[:top_edits],
        "weighted_cost_contribution_by_field": _cost_contributions(edits),
    }
    result.update({
        "semantic_preprocessing": True,
        "semantic_ted_distance": distance,
        "max_possible_cost": max_cost,
        "weighted_semantic_tree_similarity": round(weighted_similarity, 4),
        "weighted_semantic_tree_similarity_percent": round(weighted_similarity * 100, 2),
        "semantic_tree1_size": nt1.size(),
        "semantic_tree2_size": nt2.size(),
        "ignored_field_count": (
            len(debug1.get("fields_removed", [])) +
            len(debug2.get("fields_removed", []))
        ),
        "fields_normalized_count": (
            len(debug1.get("fields_normalized", [])) +
            len(debug2.get("fields_normalized", []))
        ),
        "edit_classification": {
            "expected_domain_differences": len(expected_domain),
            "structural_differences": len(structural_diffs),
            "content_updates": len(content_updates),
            "note": (
                "expected_domain_differences are normal country-specific values "
                "(population, area, GDP, dates); "
                "structural_differences indicate fields present in one country but "
                "not the other; "
                "content_updates are shared fields with differing values."
            ),
        },
        "debug": debug,
    })
    if save_debug_path:
        try:
            debug_dir = os.path.dirname(save_debug_path)
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
            with open(save_debug_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            result["debug_save_error"] = str(exc)
    return result


def _cost_contributions(edits):
    totals = {}
    for op in edits:
        path = op.get("path") or "unknown"
        canonical = _canonical_path(path)
        category = ".".join(canonical.split(".")[:2]) if "." in canonical else canonical
        totals[category] = round(totals.get(category, 0.0) + get_field_weight(canonical), 4)
    return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))


def _content_similarity(t1, t2):
    """Compare useful leaf values only for structurally matching fields."""
    f1, f2 = _field_values(t1), _field_values(t2)
    common = {k for k in set(f1) & set(f2) if not _skip_content_field(k)}
    if not common:
        return 0.0
    sims = []
    for k in common:
        sims.append(_field_similarity(k, f1[k], f2[k]))
    return sum(sims) / len(sims)


IDENTITY_CONTENT_FIELDS = {
    "name",
    "infobox_title",
    "capital",
    "capital_and_largest_city",
    "largest_city",
    "demonym",
    "currency",
    "calling_code",
    "iso_3166_code",
    "internet_tld",
    "date_format",
    "driving_side",
}

LEADER_FIELD_TERMS = {
    "president",
    "prime_minister",
    "supreme_leader",
    "monarch",
    "king",
    "queen",
    "governor_general",
    "chief_justice",
    "speaker",
    "parliament_speaker",
    "council_president",
    "assembly_president",
}


def _skip_content_field(path):
    parts = path.lower().split(".")
    leaf = parts[-1]
    if leaf in IDENTITY_CONTENT_FIELDS or leaf in LEADER_FIELD_TERMS:
        return True
    return any(p in LEADER_FIELD_TERMS for p in parts)


def _field_similarity(path, values1, values2):
    p = path.lower()
    if _is_date_field(p):
        return _date_similarity(values1, values2)
    if _is_enum_field(p):
        return _enum_similarity(values1, values2)
    if "time_zone" in p or "summer_dst" in p:
        return _timezone_similarity(values1, values2)
    if _is_numeric_field(p):
        return _numeric_similarity(values1, values2)
    if _is_distribution_field(p):
        return _distribution_similarity(values1, values2)
    if _is_set_field(p):
        return _set_similarity(values1, values2)
    if "government" in p:
        return _government_similarity(values1, values2)
    return SequenceMatcher(None, " ".join(values1), " ".join(values2)).ratio()


def _is_numeric_field(path):
    numeric_terms = (
        "area", "population", "density", "gdp", "per_capita",
        "gini", "hdi", "water", "estimate", "total"
    )
    return any(term in path for term in numeric_terms)


def _is_distribution_field(path):
    return "religion" in path or "ethnic_group" in path


def _is_set_field(path):
    set_terms = (
        "language", "vernacular", "legislature", "upper_house",
        "lower_house"
    )
    return any(term in path for term in set_terms)


_DATE_FIELD_TERMS = frozenset({
    "establishment", "independence", "founded", "formation",
    "history", "declared", "recognized", "ratified",
})


def _is_date_field(path):
    return any(t in path for t in _DATE_FIELD_TERMS)


def _extract_year(text):
    m = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)
    return int(m.group(1)) if m else None


def _date_similarity(values1, values2):
    y1 = _extract_year(" ".join(values1))
    y2 = _extract_year(" ".join(values2))
    if y1 is None or y2 is None:
        return SequenceMatcher(None, " ".join(values1), " ".join(values2)).ratio()
    if y1 == y2:
        return 1.0
    return max(0.0, 1.0 - abs(y1 - y2) / 500.0)


_ENUM_FIELD_TERMS = frozenset({"driving_side", "date_format"})


def _is_enum_field(path):
    return any(t in path for t in _ENUM_FIELD_TERMS)


def _enum_similarity(values1, values2):
    v1 = clean_value(" ".join(values1))
    v2 = clean_value(" ".join(values2))
    return 1.0 if v1 == v2 else 0.0


# Government-form categories for semantic comparison
_GOV_CATEGORIES = {
    "republic":      frozenset({"republic", "republican"}),
    "monarchy":      frozenset({"monarchy", "kingdom", "sultanate", "emirate", "principality"}),
    "federal":       frozenset({"federal", "federation"}),
    "unitary":       frozenset({"unitary"}),
    "democracy":     frozenset({"democracy", "democratic"}),
    "presidential":  frozenset({"presidential"}),
    "parliamentary": frozenset({"parliamentary", "parliament"}),
    "socialist":     frozenset({"socialist", "communist", "marxist"}),
    "theocracy":     frozenset({"theocracy", "theocratic", "islamic republic", "islamic"}),
    "authoritarian": frozenset({"authoritarian", "autocratic"}),
    # Distinguishes constitutional/parliamentary monarchies from absolute ones
    "absolute":      frozenset({"absolute"}),
}


def _gov_categories(text):
    t = text.lower()
    return {cat for cat, terms in _GOV_CATEGORIES.items() if any(kw in t for kw in terms)}


def _normalize_tokens(values):
    text = " ".join(values).lower()
    text = re.sub(r"\d+(?:\.\d+)?%?", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    stop = {
        "and", "or", "the", "of", "other", "others", "see", "groups",
        "languages", "language", "official", "national", "recognised",
        "recognized", "minority"
    }
    return {t for t in text.split() if t and t not in stop}


def _set_similarity(values1, values2):
    s1, s2 = _normalize_tokens(values1), _normalize_tokens(values2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _government_similarity(values1, values2):
    text1, text2 = " ".join(values1), " ".join(values2)
    cats1, cats2 = _gov_categories(text1), _gov_categories(text2)
    if cats1 and cats2:
        union = cats1 | cats2
        cat_sim = len(cats1 & cats2) / len(union)
    else:
        cat_sim = SequenceMatcher(None, text1, text2).ratio()
    token_sim = _set_similarity(values1, values2)
    # Category match carries more weight than token overlap — it distinguishes
    # absolute monarchy from constitutional/parliamentary monarchy correctly.
    return 0.65 * cat_sim + 0.35 * token_sim


def _distribution_similarity(values1, values2):
    d1, d2 = _parse_distribution(values1), _parse_distribution(values2)
    if d1 and d2:
        keys = set(d1) | set(d2)
        total = sum(max(d1.get(k, 0.0), d2.get(k, 0.0)) for k in keys)
        if total > 0:
            overlap = sum(min(d1.get(k, 0.0), d2.get(k, 0.0)) for k in keys)
            return overlap / total
    return _set_similarity(values1, values2)


def _parse_distribution(values):
    dist = {}
    for value in values:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", value)
        if not m:
            continue
        pct = float(m.group(1))
        label = re.sub(r"\d+(?:\.\d+)?\s*%", " ", value.lower())
        label = re.sub(r"[^a-z0-9]+", " ", label).strip()
        label = " ".join(t for t in label.split()
                         if t not in {"and", "or", "the", "of", "other", "others"})
        if label:
            dist[label] = dist.get(label, 0.0) + pct
    return dist


def _numeric_similarity(values1, values2):
    n1, n2 = _extract_number(" ".join(values1)), _extract_number(" ".join(values2))
    if n1 is None or n2 is None:
        return SequenceMatcher(None, " ".join(values1), " ".join(values2)).ratio()
    if n1 == n2:
        return 1.0
    if n1 <= 0 or n2 <= 0:
        return 0.0
    return min(n1, n2) / max(n1, n2)


def _extract_number(text):
    m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not m:
        return None
    num = float(m.group(0).replace(",", ""))
    lower = text.lower()
    if "trillion" in lower:
        num *= 1_000_000_000_000
    elif "billion" in lower:
        num *= 1_000_000_000
    elif "million" in lower:
        num *= 1_000_000
    elif "thousand" in lower:
        num *= 1_000
    return num


def _timezone_similarity(values1, values2):
    z1, z2 = _extract_utc_offset(" ".join(values1)), _extract_utc_offset(" ".join(values2))
    if z1 is None or z2 is None:
        return SequenceMatcher(None, " ".join(values1), " ".join(values2)).ratio()
    return max(0.0, 1.0 - abs(z1 - z2) / 12.0)


def _extract_utc_offset(text):
    m = re.search(r"utc\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?", text.lower())
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    hours = int(m.group(2))
    minutes = int(m.group(3) or 0)
    return sign * (hours + minutes / 60)


def _field_values(node, prefix=""):
    fields = {}
    for c in node.children:
        if c.is_structural():
            k = f"{prefix}.{c.label}" if prefix else c.label
            direct_values = [leaf.label for leaf in c.children if leaf.is_leaf()]
            if direct_values:
                fields[k] = direct_values
            fields.update(_field_values(c, k))
    return fields


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Edit Script (diff) Recovery
# ═══════════════════════════════════════════════════════════════════════════════

OP_MATCH = "Match"
OP_INSERT = "Insert"
OP_DELETE = "Delete"
OP_UPDATE = "Update"

# Fields whose values are inherently country-specific and expected to differ.
# Finding different values in these fields is a normal domain difference, NOT
# a meaningful structural mismatch or anomaly between two countries.
_EXPECTED_DOMAIN_FIELDS = frozenset({
    "demographics.population",
    "demographics.population.density",
    "demographics.population.total",
    "demographics.population.estimate",
    "demographics.hdi",
    "demographics.gini",
    "geography.area",
    "geography.area.total",
    "geography.area.km2",
    "economy.gdp",
    "economy.gdp.ppp",
    "economy.gdp.nominal",
    "economy.gdp.total",
    "economy.gdp.per_capita",
    "history.establishment",
    "history.establishment.independence",
    "history.establishment.declared",
    "history.establishment.recognized",
})

# Fields where structural presence/absence is meaningful
_STRUCTURAL_SIGNIFICANCE_FIELDS = frozenset({
    "languages.official",
    "languages.recognized",
    "demographics.religion",
    "demographics.ethnic_groups",
    "politics.government",
    "geography.region",
    "geography.continent",
    "economy.currency",
})


def classify_edit_op(op):
    """Classify an edit operation into one of three semantic categories.

    Returns:
      "match"                    — no change
      "expected_domain_difference" — fields expected to vary between any two countries
                                    (population, area, GDP, dates, etc.)
      "structural_difference"    — a whole field category is present in one tree
                                    but absent in the other
      "content_update"           — same field exists in both trees, values differ
    """
    op_type = op.get("op", "")
    node_type = op.get("type", "")
    path = (op.get("path") or "").lower()
    canonical = _canonical_path(path) if path else ""

    if op_type == OP_MATCH:
        return "match"

    for field in _EXPECTED_DOMAIN_FIELDS:
        if canonical == field or canonical.startswith(field + ".") or field.startswith(canonical + "."):
            return "expected_domain_difference"

    if node_type == "structural" and op_type in (OP_DELETE, OP_INSERT):
        return "structural_difference"

    return "content_update"


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
    edits   = [o for o in ops if o["op"] != OP_MATCH]
    matches = [o for o in ops if o["op"] == OP_MATCH]
    se = [o for o in edits if o["type"] == "structural"]
    ce = [o for o in edits if o["type"] == "content"]

    # Classify edits if not already annotated
    expected  = [o for o in edits if o.get("classification") == "expected_domain_difference"]
    struct_d  = [o for o in edits if o.get("classification") == "structural_difference"]
    content_u = [o for o in edits if o.get("classification") == "content_update"]

    print(f"    Summary: {len(matches)} matches, {len(edits)} edits "
          f"({len(se)} structural, {len(ce)} content)")
    if expected or struct_d or content_u:
        print(f"    Classification: {len(struct_d)} structural differences, "
              f"{len(content_u)} content updates, "
              f"{len(expected)} expected domain differences")
    print()

    if se:
        print("    STRUCTURAL CHANGES")
        print("    " + "─" * 50)
        for o in se:
            tag = f"  [{o.get('classification', '')}]" if o.get("classification") else ""
            if o["op"] == OP_UPDATE:
                print(f"      UPDATE field: {o['node1']}  →  {o['node2']}{tag}")
            elif o["op"] == OP_DELETE:
                print(f"      DELETE field: {o['node1']}{tag}")
            elif o["op"] == OP_INSERT:
                print(f"      INSERT field: {o['node2']}{tag}")
        print()

    if ce:
        print("    CONTENT CHANGES")
        print("    " + "─" * 50)
        for o in ce:
            field = o.get("path", "").split(".")[-1] or "?"
            tag = f"  [{o.get('classification', '')}]" if o.get("classification") else ""
            if o["op"] == OP_UPDATE:
                print(f"      UPDATE {field}: {o['node1']}  →  {o['node2']}{tag}")
            elif o["op"] == OP_DELETE:
                print(f"      DELETE {field}: {o['node1']}{tag}")
            elif o["op"] == OP_INSERT:
                print(f"      INSERT {field}: {o['node2']}{tag}")
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
    debug_path = os.path.join(
        TREE_DIR,
        f"_debug_{n1.replace(' ', '_')}_vs_{n2.replace(' ', '_')}.json"
    )
    sem = compute_similarity(t1, t2, use_semantic_preprocessing=True,
                             save_debug_path=debug_path)
    sim = tree_similarity(t1, t2)
    print(f"    Raw TED distance                : {sem['raw_ted_distance']}")
    print(f"    Raw normalized tree similarity  : {sem['raw_tree_similarity_percent']:.2f}%")
    print(f"    Semantic TED distance           : {sem['semantic_ted_distance']}")
    print(f"    Weighted semantic tree similarity: "
          f"{sem['weighted_semantic_tree_similarity_percent']:.2f}%")
    print(f"    Raw tree sizes                  : "
          f"{sem['raw_tree1_size']} / {sem['raw_tree2_size']}")
    print(f"    Semantic tree sizes             : "
          f"{sem['semantic_tree1_size']} / {sem['semantic_tree2_size']}")
    print(f"    Ignored fields                  : {sem['ignored_field_count']}")
    print(f"    Debug saved to                  : {debug_path}")
    print(f"    WARNING: {sem['warning']}")

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
