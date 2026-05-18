/**
 * charts/dendrogram.js
 *
 * Renders a left-to-right dendrogram from HAC merge_history.
 *
 * merge_history entry format (from hierarchical.py):
 *   { step, merged: [idA, idB], new_id, distance, size }
 *
 * Leaf IDs 0..n-1 map to matrixData.labels[i].
 * Internal nodes get IDs n, n+1, … (the new_id values).
 *
 * If k > 1, the tree is partial (stops before full root).
 * A lightweight virtual root is added so D3 has one tree.
 */

const DendrogramChart = (() => {

  /* ── Public entry point ─────────────────────────────────────────────────── */

  function render(containerId, result, matrixData) {
    const el = document.getElementById(containerId);
    el.innerHTML = "";

    const history = result && result.metadata && result.metadata.merge_history;
    if (!history || history.length === 0) {
      _empty(el, "Dendrogram is only available for Hierarchical clustering.");
      return;
    }
    if (!matrixData || !matrixData.labels) {
      _empty(el, "Distance matrix not loaded — cannot decode node indices.");
      return;
    }

    const labels = matrixData.labels;   // index → country name
    const root   = _buildTree(history, labels, result);
    if (!root) { _empty(el, "Could not build dendrogram tree."); return; }

    _draw(el, root, result, labels.length);
  }

  /* ── Build tree from merge_history ─────────────────────────────────────── */

  function _buildTree(history, labels, result) {
    const n       = labels.length;
    const nodeMap = new Map();

    /* Colour per cluster (for leaf coloring) */
    const clusterOf = {};
    if (result && result.assignments) {
      result.assignments.forEach(a => { clusterOf[a.label] = a.cluster_id; });
    }

    /* Initialise leaf nodes */
    labels.forEach((lbl, i) => {
      nodeMap.set(i, {
        id:      i,
        name:    lbl,
        cluster: clusterOf[lbl] ?? -1,
        height:  0,
        isLeaf:  true,
        children: [],
      });
    });

    /* Track which IDs have not yet been merged into a parent */
    const active = new Set(labels.map((_, i) => i));

    for (const step of history) {
      const [aId, bId] = step.merged;
      const newId      = step.new_id;
      const height     = step.distance;

      const nodeA = nodeMap.get(aId);
      const nodeB = nodeMap.get(bId);
      if (!nodeA || !nodeB) continue;

      /* Inherit cluster colour from children if they agree */
      const clusterA = _dominantCluster(nodeA);
      const clusterB = _dominantCluster(nodeB);
      const cluster  = clusterA === clusterB ? clusterA : -1;

      nodeMap.set(newId, {
        id:       newId,
        name:     `node_${newId}`,
        cluster,
        height,
        isLeaf:   false,
        children: [nodeA, nodeB],
      });

      active.delete(aId);
      active.delete(bId);
      active.add(newId);
    }

    /* Remaining active nodes are the k sub-tree roots */
    const roots = [...active].map(id => nodeMap.get(id)).filter(Boolean);

    if (roots.length === 1) return roots[0];

    /* Virtual root: height slightly above the last merge */
    const maxH = history[history.length - 1].distance;
    return {
      id:       -1,
      name:     "root",
      cluster:  -1,
      height:   maxH * 1.05,
      isLeaf:   false,
      isVirtual: true,
      children: roots,
    };
  }

  /* Recursively find the single cluster if all leaves share one */
  function _dominantCluster(node) {
    if (node.isLeaf) return node.cluster;
    const set = new Set();
    _collectClusters(node, set);
    return set.size === 1 ? [...set][0] : -1;
  }
  function _collectClusters(node, set) {
    if (node.isLeaf) { set.add(node.cluster); return; }
    node.children.forEach(c => _collectClusters(c, set));
  }

  /* ── Draw ───────────────────────────────────────────────────────────────── */

  function _draw(el, root, result, n) {
    const PALETTE = [
      "#6366f1","#22c55e","#f59e0b","#ef4444","#a855f7",
      "#3b82f6","#ec4899","#14b8a6","#f97316","#84cc16",
    ];
    const color = (cluster) =>
      cluster >= 0 ? PALETTE[cluster % PALETTE.length] : "#8892a4";

    /* ── Compute leaf y-positions (depth-first left-to-right) ── */
    const leaves = [];
    (function collectLeaves(node) {
      if (node.isLeaf) { leaves.push(node); return; }
      node.children.forEach(collectLeaves);
    })(root);

    const leafCount = leaves.length;
    const ROW_H     = Math.max(18, Math.min(28, 560 / leafCount));
    const HEIGHT    = leafCount * ROW_H;
    const maxDist   = root.height;

    /* Assign y to every leaf in order */
    leaves.forEach((leaf, i) => { leaf._y = (i + 0.5) * ROW_H; });

    /* Propagate y up (average of children) */
    function assignY(node) {
      if (node.isLeaf) return node._y;
      const ys = node.children.map(assignY);
      node._y  = (Math.min(...ys) + Math.max(...ys)) / 2;
      return node._y;
    }
    assignY(root);

    /* ── SVG setup with zoom/pan ── */
    const LABEL_W = 130;
    const AXIS_H  = 36;
    const PAD     = { top: AXIS_H + 8, right: LABEL_W + 8, bottom: 16, left: 12 };
    const svgW    = (el.clientWidth  || 720);
    const svgH    = HEIGHT + PAD.top + PAD.bottom;
    const plotW   = svgW  - PAD.left - PAD.right;
    const plotH   = svgH  - PAD.top  - PAD.bottom;

    const xScale = d3.scaleLinear().domain([maxDist, 0]).range([0, plotW]);

    const svg = d3.select(el)
      .append("svg")
      .attr("width",  svgW)
      .attr("height", svgH)
      .style("display", "block");

    /* Clip path so links don't spill into the label column */
    svg.append("defs").append("clipPath").attr("id", "dend-clip")
      .append("rect").attr("width", plotW).attr("height", plotH + 4).attr("y", -2);

    const zoom = d3.zoom()
      .scaleExtent([0.5, 6])
      .on("zoom", ({ transform }) => { zoomG.attr("transform", transform); });

    svg.call(zoom);

    const outerG  = svg.append("g")
      .attr("transform", `translate(${PAD.left},${PAD.top})`);

    const zoomG   = outerG.append("g");   // this group zooms
    const linksG  = zoomG.append("g").attr("clip-path", "url(#dend-clip)");
    const nodesG  = zoomG.append("g");

    /* ── Draw links ── */
    function drawLinks(node) {
      if (!node.children || node.children.length === 0) return;

      const px = xScale(node.height);   // parent x (further right = lower distance)
      const py = node._y;

      /* Vertical bar at parent x connecting the children */
      const ys = node.children.map(c => c._y);
      const yMin = Math.min(...ys);
      const yMax = Math.max(...ys);

      linksG.append("line")
        .attr("x1", px).attr("y1", yMin)
        .attr("x2", px).attr("y2", yMax)
        .attr("stroke", node.isVirtual ? "#2e3350" : color(_dominantCluster(node)))
        .attr("stroke-width", node.isVirtual ? 1 : 1.5)
        .attr("stroke-opacity", node.isVirtual ? 0.4 : 0.7);

      /* Horizontal arms to each child */
      node.children.forEach(child => {
        const cx = xScale(child.height);
        const cy = child._y;
        const lineColor = child.isLeaf
          ? color(child.cluster)
          : (node.isVirtual ? "#2e3350" : color(_dominantCluster(child)));

        linksG.append("line")
          .attr("x1", px).attr("y1", cy)
          .attr("x2", cx).attr("y2", cy)
          .attr("stroke", lineColor)
          .attr("stroke-width", child.isLeaf ? 1 : 1.5)
          .attr("stroke-opacity", node.isVirtual ? 0.4 : 0.8);

        drawLinks(child);
      });
    }
    drawLinks(root);

    /* ── Draw leaf nodes and labels ── */
    leaves.forEach(leaf => {
      const x = xScale(0);
      const y = leaf._y;
      const c = color(leaf.cluster);

      nodesG.append("circle")
        .attr("cx", x).attr("cy", y).attr("r", 3.5)
        .attr("fill", c)
        .attr("stroke", "#0f1117")
        .attr("stroke-width", 1);

      nodesG.append("text")
        .attr("x", x + 6)
        .attr("y", y + 4)
        .attr("font-size", `${Math.max(9, ROW_H * 0.52)}px`)
        .attr("fill", c)
        .text(cleanLabel(leaf.name));
    });

    /* ── Distance axis ── */
    const axisScale = d3.scaleLinear()
      .domain([0, maxDist])
      .range([plotW, 0]);   // reversed: 0 on right, max on left

    const axis = d3.axisTop(axisScale)
      .ticks(6)
      .tickFormat(d3.format(".3f"));

    const axisG = outerG.append("g").call(axis);
    axisG.select(".domain").attr("stroke", "#2e3350");
    axisG.selectAll("line").attr("stroke", "#2e3350");
    axisG.selectAll("text")
      .attr("fill", "#8892a4")
      .attr("font-size", "10px");
    axisG.append("text")
      .attr("x", plotW / 2)
      .attr("y", -24)
      .attr("text-anchor", "middle")
      .attr("fill", "#8892a4")
      .attr("font-size", "10px")
      .text("Merge distance (lower = more similar)   ←");

    /* ── Zoom hint ── */
    outerG.append("text")
      .attr("x", svgW - PAD.left - PAD.right)
      .attr("y", plotH + PAD.bottom - 2)
      .attr("text-anchor", "end")
      .attr("font-size", "9px")
      .attr("fill", "#3d4463")
      .text("scroll to zoom · drag to pan");
  }

  /* ── Helpers ─────────────────────────────────────────────────────────────── */

  function _empty(el, msg) {
    el.innerHTML = `<div class="empty-state"><div class="empty-text">${msg}</div></div>`;
  }

  return { render };
})();
