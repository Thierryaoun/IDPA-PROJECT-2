/**
 * charts/clustermap.js
 * Force-directed graph where nodes are countries, coloured by cluster.
 * Edge strength is proportional to similarity (1 - distance).
 */

const ClustermapChart = (() => {

  function render(containerId, result, matrixData) {
    const el = document.getElementById(containerId);
    el.innerHTML = "";

    if (!result || !result.assignments || !matrixData) {
      el.innerHTML = '<div class="empty-state"><div class="empty-text">Load a result to see the cluster map.</div></div>';
      return;
    }

    const W = el.clientWidth || 680;
    const H = 460;

    const palette = ["#4f46e5","#16a34a","#d97706","#dc2626","#7c3aed",
                     "#2563eb","#db2777","#0d9488","#ea580c","#65a30d"];

    const clusterOf = {};
    const isMedoid = {};
    result.assignments.forEach(a => {
      clusterOf[a.label] = a.cluster_id;
      if (a.label === a.medoid) isMedoid[a.label] = true;
    });

    const indexMap = {};
    matrixData.labels.forEach((lbl, i) => { indexMap[lbl] = i; });

    const nodes = result.assignments.map(a => ({
      id: a.label,
      cluster: a.cluster_id,
      medoid: isMedoid[a.label] || false,
    }));

    // Build edges — only include strong similarity links to avoid clutter
    const links = [];
    const threshold = (matrixData.stats ? matrixData.stats.mean * 0.6 : 0.5);
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const ni = nodes[i].id;
        const nj = nodes[j].id;
        const oi = indexMap[ni];
        const oj = indexMap[nj];
        if (oi === undefined || oj === undefined) continue;
        const dist = matrixData.matrix[oi][oj];
        if (dist < threshold) {
          links.push({ source: ni, target: nj, dist });
        }
      }
    }

    const svg = d3.select(el)
      .append("svg")
      .attr("width", W)
      .attr("height", H)
      .style("background", "transparent");

    // Cluster hulls
    const hullG = svg.append("g").attr("class", "hulls");

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(d => d.dist * 120 + 30).strength(0.3))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide(18));

    const linkG = svg.append("g");
    const nodeG = svg.append("g");
    const labelG = svg.append("g");

    const linkSel = linkG.selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", "#cbd5e1")
      .attr("stroke-width", d => Math.max(0.5, (1 - d.dist) * 2))
      .attr("stroke-opacity", 0.7);

    const nodeSel = nodeG.selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", d => d.medoid ? 9 : 6)
      .attr("fill", d => palette[d.cluster % palette.length])
      .attr("stroke", d => d.medoid ? "#1a2035" : "#ffffff")
      .attr("stroke-width", d => d.medoid ? 2 : 1.5)
      .call(drag(sim));

    nodeSel.append("title").text(d => `${d.id} (cluster ${d.cluster})`);

    const labelSel = labelG.selectAll("text")
      .data(nodes)
      .join("text")
      .attr("font-size", "10px")
      .attr("fill", d => palette[d.cluster % palette.length])
      .attr("text-anchor", "middle")
      .attr("dy", "-0.8em")
      .text(d => d.id)
      .style("pointer-events", "none");

    sim.on("tick", () => {
      linkSel
        .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      nodeSel.attr("cx", d => d.x).attr("cy", d => d.y);
      labelSel.attr("x", d => d.x).attr("y", d => d.y);

      // Draw convex hulls per cluster
      const byCluster = d3.group(nodes, d => d.cluster);
      hullG.selectAll("path").remove();
      byCluster.forEach((pts, cid) => {
        if (pts.length < 3) return;
        const hull = d3.polygonHull(pts.map(p => [p.x, p.y]));
        if (!hull) return;
        hullG.append("path")
          .attr("d", "M" + hull.map(p => p.join(",")).join("L") + "Z")
          .attr("fill", palette[cid % palette.length])
          .attr("fill-opacity", 0.07)
          .attr("stroke", palette[cid % palette.length])
          .attr("stroke-opacity", 0.25)
          .attr("stroke-width", 1.5);
      });
    });

    function drag(simulation) {
      return d3.drag()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        });
    }
  }

  return { render };
})();
