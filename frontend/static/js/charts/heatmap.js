/**
 * charts/heatmap.js
 *
 * Distance matrix heatmap, sorted by cluster membership.
 * Cells are coloured by TED distance (0 = identical, 1 = maximally different).
 * Hover shows a tooltip with both country names and the exact distance value.
 * Cluster boundaries are highlighted with coloured separator lines.
 */

const HeatmapChart = (() => {

  /* ── Public entry point ─────────────────────────────────────────────────── */

  function render(containerId, result, matrixData) {
    const el = document.getElementById(containerId);
    el.innerHTML = "";

    if (!matrixData || !matrixData.labels) {
      _empty(el, "Distance matrix not loaded.");
      return;
    }

    const PALETTE = [
      "#6366f1","#22c55e","#f59e0b","#ef4444","#a855f7",
      "#3b82f6","#ec4899","#14b8a6","#f97316","#84cc16",
    ];

    /* ── Build cluster-sorted label order ── */
    const clusterOf = {};
    if (result && result.assignments) {
      result.assignments.forEach(a => { clusterOf[a.label] = a.cluster_id; });
    }

    const allLabels  = matrixData.labels;
    const n          = allLabels.length;
    const origIndex  = {};
    allLabels.forEach((lbl, i) => { origIndex[lbl] = i; });

    /* Sort labels: by cluster id first, then alphabetically within cluster */
    const sortedLabels = [...allLabels].sort((a, b) => {
      const ca = clusterOf[a] ?? 999;
      const cb = clusterOf[b] ?? 999;
      return ca !== cb ? ca - cb : a.localeCompare(b);
    });

    /* ── Layout ── */
    const containerW = el.clientWidth || 720;

    /* Decide label area and cell size based on n */
    const showLabels = n <= 60;
    const LABEL_AREA = showLabels ? Math.min(100, Math.max(60, Math.floor(120 * (30 / n)))) : 0;
    const MAX_CELL   = showLabels ? Math.floor((containerW - LABEL_AREA - 40) / n) : Math.floor((containerW - 40) / n);
    const CELL       = Math.max(4, Math.min(MAX_CELL, 20));

    const plotW   = n * CELL;
    const plotH   = n * CELL;
    const PAD     = { top: LABEL_AREA + 16, right: 16, bottom: 50, left: LABEL_AREA + 16 };
    const totalW  = plotW + PAD.left + PAD.right;
    const totalH  = plotH + PAD.top  + PAD.bottom;

    /* ── Color scale ── */
    const maxDist = matrixData.stats ? matrixData.stats.max : 1;
    const colorScale = d3.scaleSequential()
      .domain([0, maxDist])
      .interpolator(d3.interpolateInferno)
      .clamp(true);

    /* ── SVG ── */
    const svg = d3.select(el)
      .append("svg")
      .attr("width",  totalW)
      .attr("height", totalH)
      .style("display", "block")
      .style("overflow", "visible");

    const g = svg.append("g")
      .attr("transform", `translate(${PAD.left},${PAD.top})`);

    /* ── Tooltip (HTML overlay) ── */
    let tooltip = d3.select(el).select(".hm-tooltip");
    if (tooltip.empty()) {
      tooltip = d3.select(el)
        .append("div")
        .attr("class", "hm-tooltip")
        .style("position", "absolute")
        .style("background",  "rgba(255,255,255,0.97)")
        .style("border",      "1px solid #bcc6d4")
        .style("border-radius", "6px")
        .style("padding",     "7px 11px")
        .style("font-size",   "12px")
        .style("color",       "#1a2035")
        .style("pointer-events", "none")
        .style("white-space", "nowrap")
        .style("z-index",     "50")
        .style("box-shadow",  "0 4px 12px rgba(0,0,0,0.10)")
        .style("display",     "none");
    }

    /* ── Draw cells ── */
    sortedLabels.forEach((rowLbl, ri) => {
      const origR = origIndex[rowLbl];
      sortedLabels.forEach((colLbl, ci) => {
        const origC = origIndex[colLbl];
        const dist  = matrixData.matrix[origR][origC];
        const isDiag = origR === origC;

        g.append("rect")
          .attr("x",      ci * CELL)
          .attr("y",      ri * CELL)
          .attr("width",  CELL)
          .attr("height", CELL)
          .attr("fill",   isDiag ? "#e8edf4" : colorScale(dist))
          .attr("stroke", CELL > 6 ? "#ffffff" : "none")
          .attr("stroke-width", CELL > 6 ? 0.5 : 0)
          .on("mouseover", function(event) {
            if (!isDiag) {
              const cR = clusterOf[rowLbl];
              const cC = clusterOf[colLbl];
              const cRColor = cR !== undefined ? PALETTE[cR % PALETTE.length] : "#8892a4";
              const cCColor = cC !== undefined ? PALETTE[cC % PALETTE.length] : "#8892a4";
              tooltip
                .style("display", "block")
                .html(`
                  <div style="margin-bottom:4px">
                    <span style="color:${cRColor};font-weight:700">${cleanLabel(rowLbl)}</span>
                    <span style="color:#5a6a85"> vs </span>
                    <span style="color:${cCColor};font-weight:700">${cleanLabel(colLbl)}</span>
                  </div>
                  <div>Distance: <strong style="color:#1a2035">${dist.toFixed(4)}</strong></div>
                  <div style="color:#5a6a85;font-size:10px">Similarity: ${((1-dist)*100).toFixed(1)}%</div>
                `);
            }
            d3.select(this)
              .attr("stroke", "#ffffff")
              .attr("stroke-width", 1.5);
          })
          .on("mousemove", function(event) {
            const rect = el.getBoundingClientRect();
            const x    = event.clientX - rect.left + 14;
            const y    = event.clientY - rect.top  - 10;
            tooltip
              .style("left", `${x}px`)
              .style("top",  `${y}px`);
          })
          .on("mouseout", function() {
            tooltip.style("display", "none");
            const origR2 = origIndex[rowLbl];
            const origC2 = origIndex[colLbl];
            d3.select(this)
              .attr("stroke", CELL > 6 ? "#ffffff" : "none")
              .attr("stroke-width", CELL > 6 ? 0.5 : 0);
          });
      });
    });

    /* ── Cluster boundary separators ── */
    if (result && result.clusters) {
      const cids = Object.keys(result.clusters).map(Number).sort((a, b) => a - b);
      let pos = 0;
      cids.forEach(cid => {
        const members = result.clusters[cid];
        /* Count how many sorted labels are in this cluster */
        const cnt = sortedLabels.filter(l => clusterOf[l] === cid).length;
        pos += cnt;
        if (pos < n) {
          const px = pos * CELL;
          const c  = PALETTE[cid % PALETTE.length];
          /* Vertical line */
          g.append("line")
            .attr("x1", px).attr("y1", 0)
            .attr("x2", px).attr("y2", plotH)
            .attr("stroke", c).attr("stroke-width", 1.5).attr("stroke-opacity", 0.6);
          /* Horizontal line */
          g.append("line")
            .attr("x1", 0).attr("y1", px)
            .attr("x2", plotW).attr("y2", px)
            .attr("stroke", c).attr("stroke-width", 1.5).attr("stroke-opacity", 0.6);
        }
      });

      /* Cluster label boxes above and to the left */
      if (showLabels) {
        let pos2 = 0;
        cids.forEach(cid => {
          const cnt    = sortedLabels.filter(l => clusterOf[l] === cid).length;
          const mid    = (pos2 + cnt / 2) * CELL;
          const c      = PALETTE[cid % PALETTE.length];
          /* Top bracket */
          g.append("text")
            .attr("x",           mid)
            .attr("y",           -LABEL_AREA - 4)
            .attr("text-anchor", "middle")
            .attr("font-size",   "10px")
            .attr("fill",        c)
            .attr("font-weight", "700")
            .text(`C${cid}`);
          pos2 += cnt;
        });
      }
    }

    /* ── Row/column labels ── */
    if (showLabels) {
      const fontSize = Math.max(8, Math.min(CELL - 1, 11));
      sortedLabels.forEach((lbl, i) => {
        const cid = clusterOf[lbl];
        const c   = cid !== undefined ? PALETTE[cid % PALETTE.length] : "#8892a4";
        const mid = i * CELL + CELL / 2;
        const displayName = cleanLabel(lbl);

        /* Left row label */
        g.append("text")
          .attr("x",           -6)
          .attr("y",           mid + fontSize * 0.35)
          .attr("text-anchor", "end")
          .attr("font-size",   `${fontSize}px`)
          .attr("fill",        c)
          .text(displayName.length > 14 ? displayName.slice(0, 13) + "…" : displayName);

        /* Top column label (rotated) */
        g.append("text")
          .attr("transform",   `rotate(-60, ${mid}, ${-6})`)
          .attr("x",           mid)
          .attr("y",           -6)
          .attr("text-anchor", "start")
          .attr("font-size",   `${fontSize}px`)
          .attr("fill",        c)
          .text(displayName.length > 14 ? displayName.slice(0, 13) + "…" : displayName);
      });
    } else {
      /* For large matrices: show a count badge instead */
      g.append("text")
        .attr("x",           0)
        .attr("y",           -8)
        .attr("font-size",   "11px")
        .attr("fill",        "#64748b")
        .text(`${n} x ${n} matrix  (labels hidden above 60 countries — hover for names)`);
    }

    /* ── Colour legend ── */
    const lgW    = Math.min(220, plotW);
    const lgX    = (plotW - lgW) / 2;
    const lgY    = plotH + 16;

    const defs = svg.append("defs");
    const grad = defs.append("linearGradient").attr("id", "hm-gradient-" + containerId);
    const stops = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    stops.forEach(t => {
      grad.append("stop")
        .attr("offset", `${t * 100}%`)
        .attr("stop-color", colorScale(t * maxDist));
    });

    const lg = g.append("g").attr("transform", `translate(${lgX},${lgY})`);
    lg.append("rect")
      .attr("width", lgW).attr("height", 10).attr("rx", 3)
      .attr("fill", `url(#hm-gradient-${containerId})`);
    lg.append("text").attr("y", 22).attr("font-size", "10px").attr("fill", "#64748b")
      .text("0 (identical)");
    lg.append("text").attr("x", lgW).attr("y", 22)
      .attr("text-anchor", "end").attr("font-size", "10px").attr("fill", "#64748b")
      .text(`${maxDist.toFixed(3)} (max distance)`);
    lg.append("text").attr("x", lgW / 2).attr("y", 34)
      .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "#94a3b8")
      .text("TED distance");
  }

  /* ── Helpers ─────────────────────────────────────────────────────────────── */

  function _empty(el, msg) {
    el.innerHTML = `<div class="empty-state"><div class="empty-text">${msg}</div></div>`;
  }

  return { render };
})();
