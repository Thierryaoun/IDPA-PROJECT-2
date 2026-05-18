/**
 * app.js — Router, global state, and page controllers.
 */

/* ── Region quick-select map ───────────────────────────────────────────────── */
const REGION_MAP = {
  MiddleEast: ["Lebanon","Syrian_Arab_Republic","Jordan","Iraq","Egypt","Saudi_Arabia","Iran__Islamic_Republic_of","Israel","Yemen","Kuwait","Bahrain","Qatar","Oman"],
  Europe:     ["France","Germany","Switzerland","Italy","Spain","United_Kingdom","Poland","Netherlands","Sweden","Norway","Denmark","Finland","Belgium","Austria","Portugal","Greece","Czech_Republic","Hungary","Romania","Bulgaria"],
  LatAm:      ["Brazil","Argentina","Chile","Colombia","Peru","Mexico","Venezuela__Bolivarian_Republic_of","Ecuador","Bolivia__Plurinational_State_of","Paraguay","Uruguay","Cuba","Dominican_Republic","Guatemala","Honduras","El_Salvador","Nicaragua","Costa_Rica","Panama"],
  Asia:       ["China","Japan","India","Republic_of_Korea","Indonesia","Philippines","Viet_Nam","Thailand","Malaysia","Singapore","Bangladesh","Pakistan","Sri_Lanka","Nepal","Myanmar","Cambodia","Lao_People_s_Democratic_Republic","Mongolia"],
  Africa:     ["Nigeria","Kenya","Ethiopia","South_Africa","Ghana","United_Republic_of_Tanzania","Uganda","Rwanda","Senegal","Morocco","Algeria","Tunisia","Libya","Sudan","Egypt","Cameroon","Cote_d_Ivoire","Mali","Niger"],
};

/* ── Label cleaning ─────────────────────────────────────────────────────────
   Convert raw filesystem stems to readable country names.
   "Antigua_and_Barbuda"  →  "Antigua and Barbuda"
   "Bahamas__The"         →  "Bahamas, The"
   "Syrian_Arab_Republic" →  "Syrian Arab Republic"
────────────────────────────────────────────────────────────────────────────── */
function cleanLabel(name) {
  if (!name) return name;
  return name
    .replace(/__/g, ", ")   // double underscore → ", "
    .replace(/_/g,  " ")    // single underscore → space
    .trim();
}

/* ── App singleton ──────────────────────────────────────────────────────────── */
const App = (() => {
  /* State */
  let _allCountries  = [];   // raw IDs (file stems, no .json)
  let _selected      = new Set();
  let _currentResult = null;
  let _matrixData    = null;
  let _savedResults  = [];
  let _currentAlgo   = "kmedoids";
  let _pollTimer     = null;
  let _activeTab     = "clusters";

  /* ── Init ─────────────────────────────────────────────────────────────── */
  async function init() {
    document.querySelectorAll(".nav-item").forEach(item => {
      item.addEventListener("click", e => {
        e.preventDefault();
        navigate(item.dataset.page);
      });
    });

    document.getElementById("country-search").addEventListener("input", e => {
      renderCountryGrid(e.target.value.toLowerCase());
    });

    await refreshStatus();
    await loadCountries();
    await refreshResults();
    setInterval(refreshStatus, 8000);
  }

  /* ── Navigation ───────────────────────────────────────────────────────── */
  function navigate(page) {
    document.querySelectorAll(".nav-item").forEach(el =>
      el.classList.toggle("active", el.dataset.page === page));
    document.querySelectorAll(".page").forEach(el =>
      el.classList.toggle("active", el.id === `page-${page}`));
    if (page === "results") refreshResults();
    if (page === "compare") refreshResults();
  }

  /* ── Status indicator ─────────────────────────────────────────────────── */
  async function refreshStatus() {
    try {
      const s = await API.getStatus();
      const dot   = document.getElementById("status-dot");
      const label = document.getElementById("status-label");
      if (s.matrix_loaded) {
        dot.className = "status-dot online";
        label.textContent = `Matrix ${s.n_loaded}×${s.n_loaded}`;
      } else if (s.matrix_cached) {
        dot.className = "status-dot loading";
        label.textContent = "Cache ready";
      } else {
        dot.className = "status-dot offline";
        label.textContent = "Not ready";
      }
      document.getElementById("stat-matrix").textContent =
        s.matrix_loaded ? `${s.n_loaded}×${s.n_loaded}`
        : s.matrix_cached ? "Cached" : "—";
    } catch (_) {}
  }

  /* ── Country loading ──────────────────────────────────────────────────── */
  async function loadCountries() {
    try {
      const data = await API.getCountries();
      _allCountries = data.countries;
      _selected = new Set(_allCountries);
      document.getElementById("stat-available").textContent = _allCountries.length;
      renderCountryGrid();
      updateSelCount();
    } catch (e) {
      toast("Could not load countries: " + e.message, "error");
    }
  }

  function renderCountryGrid(filter = "") {
    const grid = document.getElementById("country-grid");
    grid.innerHTML = "";
    _allCountries.forEach(name => {
      const displayName = cleanLabel(name);
      if (filter && !displayName.toLowerCase().includes(filter)) return;
      const chip = document.createElement("div");
      chip.className = "country-chip" + (_selected.has(name) ? " selected" : "");
      chip.innerHTML = `<span class="chip-check"></span><span>${displayName}</span>`;
      chip.addEventListener("click", () => {
        if (_selected.has(name)) _selected.delete(name);
        else _selected.add(name);
        chip.classList.toggle("selected", _selected.has(name));
        updateSelCount();
      });
      grid.appendChild(chip);
    });
  }

  function updateSelCount() {
    document.getElementById("stat-selected").textContent = _selected.size;
  }

  /* ── Region quick-select ──────────────────────────────────────────────── */
  function selectRegion(region) {
    if (region === "all")        { _allCountries.forEach(n => _selected.add(n)); }
    else if (region === "none")  { _selected.clear(); }
    else {
      const list = REGION_MAP[region] || [];
      list.forEach(n => { if (_allCountries.includes(n)) _selected.add(n); });
    }
    renderCountryGrid(document.getElementById("country-search").value.toLowerCase());
    updateSelCount();
  }

  /* ── Matrix computation ───────────────────────────────────────────────── */
  async function computeMatrix() {
    const useS    = document.getElementById("use-semantic").checked;
    const force   = document.getElementById("force-recompute").checked;
    const countries = _selected.size === _allCountries.length ? null : [..._selected];

    if (_selected.size < 2) {
      toast("Select at least 2 countries.", "error"); return;
    }

    document.getElementById("matrix-progress").classList.remove("hidden");
    setBar(0, "Starting…");
    document.getElementById("btn-compute").disabled = true;

    try {
      const { job_id } = await API.startMatrix({ countries, use_semantic: useS, force });
      pollJob(job_id, async () => {
        setBar(100, "Done.");
        await refreshStatus();
        try { _matrixData = await API.getMatrix(); } catch (_) {}
        toast("Distance matrix ready.", "success");
        document.getElementById("btn-compute").disabled = false;
      });
    } catch (e) {
      toast(e.message, "error");
      document.getElementById("btn-compute").disabled = false;
    }
  }

  function setBar(pct, msg) {
    document.getElementById("matrix-bar").style.width = pct + "%";
    document.getElementById("matrix-msg").textContent  = msg;
  }

  function pollJob(jobId, onDone) {
    if (_pollTimer) clearInterval(_pollTimer);
    _pollTimer = setInterval(async () => {
      try {
        const job = await API.pollJob(jobId);
        setBar(job.progress || 0, job.message || "");
        if (job.status === "done")  { clearInterval(_pollTimer); onDone(); }
        if (job.status === "error") {
          clearInterval(_pollTimer);
          toast("Error: " + job.error, "error");
          document.getElementById("btn-compute").disabled = false;
        }
      } catch (_) {}
    }, 1200);
  }

  /* ── Algorithm selection ──────────────────────────────────────────────── */
  function selectAlgo(algo) {
    _currentAlgo = algo;
    document.querySelectorAll(".algo-card").forEach(c => {
      c.classList.toggle("selected", c.dataset.algo === algo);
    });
    document.getElementById("km-params").classList.toggle("hidden",  algo !== "kmedoids");
    document.getElementById("hac-params").classList.toggle("hidden", algo !== "hierarchical");
  }

  /* ── Run clustering ───────────────────────────────────────────────────── */
  async function runClustering() {
    const k       = parseInt(document.getElementById("param-k").value);
    const init    = document.getElementById("param-init").value;
    const maxIter = parseInt(document.getElementById("param-iter").value);
    const linkage = document.getElementById("param-linkage").value;

    const spinner = document.getElementById("run-spinner");
    const btn     = document.getElementById("btn-run");
    spinner.classList.remove("hidden");
    btn.disabled = true;

    try {
      const res = await API.runCluster({
        algorithm: _currentAlgo, n_clusters: k,
        linkage, init_method: init, max_iter: maxIter,
      });
      _currentResult = res.result;
      toast(`Clustering done — ${res.result_file}`, "success");
      await refreshResults();
      navigate("results");
      await loadResultData(res.result_file);
    } catch (e) {
      toast(e.message, "error");
    } finally {
      spinner.classList.add("hidden");
      btn.disabled = false;
    }
  }

  /* ── Results page ─────────────────────────────────────────────────────── */
  async function refreshResults() {
    try {
      const data = await API.listResults();
      _savedResults = data.results;
      const sel = document.getElementById("result-select");
      const cur = sel.value;
      sel.innerHTML = '<option value="">— select a saved result —</option>';
      data.results.forEach(r => {
        const opt       = document.createElement("option");
        opt.value       = r.filename;
        const sil       = r.silhouette_score !== null
          ? `sil=${r.silhouette_score.toFixed(3)}`  : "no sil";
        opt.textContent = `${r.filename}  (${sil})`;
        sel.appendChild(opt);
      });
      if (cur && data.results.find(r => r.filename === cur)) sel.value = cur;

      /* Populate compare selects */
      ["cmp-a", "cmp-b"].forEach(id => {
        const s = document.getElementById(id);
        const v = s.value;
        s.innerHTML = '<option value="">— select —</option>';
        data.results.forEach(r => {
          const o = document.createElement("option");
          o.value = r.filename;
          o.textContent = r.filename;
          s.appendChild(o);
        });
        if (v) s.value = v;
      });
    } catch (_) {}
  }

  async function loadResult(filename) {
    if (!filename) return;
    await loadResultData(filename);
  }

  async function loadResultData(filename) {
    try {
      const data = await API.getResult(filename);
      _currentResult = data;

      /* Load the matrix that produced this result (by ID if available, size otherwise) */
      const matrixId = data.metadata && data.metadata.matrix_id;
      const nDocs    = data.metadata && data.metadata.n_documents;
      if (matrixId) {
        if (!_matrixData || _matrixData.matrix_id !== matrixId) {
          try { _matrixData = await API.getMatrix({ id: matrixId }); } catch (_) {}
        }
      } else if (nDocs && (!_matrixData || _matrixData.n !== nDocs)) {
        try { _matrixData = await API.getMatrix({ n: nDocs }); } catch (_) {}
      }

      renderResultMetrics(data);
      renderClusterCards(data);
      renderCurrentTab();

      document.getElementById("metric-cards").style.display  = "";
      document.getElementById("result-body").style.display   = "";
      document.getElementById("results-empty").style.display = "none";

      /* Auto-switch tab to clusters */
      _activeTab = "clusters";
      document.querySelectorAll(".tab-btn").forEach(b =>
        b.classList.toggle("active", b.dataset.tab === "clusters"));
      document.querySelectorAll(".tab-pane").forEach(p =>
        p.classList.toggle("active", p.id === "tab-clusters"));
    } catch (e) {
      toast("Could not load result: " + e.message, "error");
    }
  }

  function renderResultMetrics(result) {
    const algo = result.algorithm === "hierarchical"
      ? `HAC (${result.linkage || "avg"})` : "K-Medoids";
    document.getElementById("m-algo").textContent    = algo;
    document.getElementById("m-k").textContent       = result.n_clusters;
    document.getElementById("m-sil").textContent     =
      result.silhouette_score !== null ? result.silhouette_score.toFixed(3) : "n/a";
    document.getElementById("m-inertia").textContent =
      result.inertia !== null ? result.inertia.toFixed(3) : "n/a";
  }

  function renderClusterCards(result) {
    const container = document.getElementById("cluster-cards-container");
    container.innerHTML = "";
    const PALETTE = [
      "#4f46e5","#16a34a","#d97706","#dc2626","#7c3aed",
      "#2563eb","#db2777","#0d9488","#ea580c","#65a30d",
    ];

    const distOf = {};
    result.assignments.forEach(a => { distOf[a.label] = a.distance_to_medoid; });

    Object.keys(result.clusters).map(Number).sort((a, b) => a - b).forEach(cid => {
      const members = result.clusters[cid];
      const medoid  = result.medoids[cid];
      const stats   = result.cluster_stats ? result.cluster_stats[cid] : null;
      const c       = PALETTE[cid % PALETTE.length];

      const card    = document.createElement("div");
      card.className = "cluster-card";
      card.style.cssText = `border-left: 3px solid ${c}`;

      card.innerHTML = `
        <div class="cluster-card-header">
          <span class="cluster-badge" style="background:${c}1a;color:${c}">
            Cluster ${cid}&nbsp;&bull;&nbsp;${members.length} members
          </span>
          <span class="cluster-medoid">Medoid:&nbsp;<strong>${cleanLabel(medoid)}</strong></span>
        </div>
        <div class="cluster-member-list">
          ${members.slice().sort().map(name => `
            <div class="cluster-member${name === medoid ? " is-medoid" : ""}">
              <span>${name === medoid ? "&#9670;&nbsp;" : ""}${cleanLabel(name)}</span>
              <span class="member-dist">${(distOf[name] ?? 0).toFixed(4)}</span>
            </div>
          `).join("")}
        </div>
        ${stats ? `
        <div class="cluster-stats-bar">
          <span class="cstat">avg dist&nbsp;<strong>${(stats.avg_distance_to_medoid||0).toFixed(3)}</strong></span>
          <span class="cstat">pairwise&nbsp;<strong>${(stats.avg_pairwise_distance||0).toFixed(3)}</strong></span>
          <span class="cstat">diam&nbsp;<strong>${(stats.diameter||0).toFixed(3)}</strong></span>
        </div>` : ""}
      `;
      container.appendChild(card);
    });
  }

  /* ── Tab switching ────────────────────────────────────────────────────── */
  function switchTab(tab, btn) {
    _activeTab = tab;
    document.querySelectorAll(".tab-btn").forEach(b =>
      b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab-pane").forEach(p =>
      p.classList.toggle("active", p.id === `tab-${tab}`));
    renderCurrentTab();
  }

  function renderCurrentTab() {
    if (!_currentResult) return;
    if (_activeTab === "clusters") return;  // cluster cards rendered separately

    const chartId = `chart-${_activeTab}`;
    const chartEl = document.getElementById(chartId);
    if (!chartEl) return;

    /* Check matrix-result provenance */
    const resultMid = _currentResult.metadata && _currentResult.metadata.matrix_id;
    const hasInfo   = !!resultMid;

    if (!_matrixData) {
      chartEl.innerHTML = '<div class="empty-state"><div class="empty-text">Distance matrix not loaded. Use the Dataset page to compute or load a matrix first.</div></div>';
      return;
    }

    if (hasInfo && _matrixData.matrix_id && _matrixData.matrix_id !== resultMid) {
      chartEl.innerHTML = `<div class="empty-state warn"><div class="empty-text">
        <strong>Matrix mismatch.</strong><br>
        This result was computed with matrix&nbsp;<code>${resultMid}</code>
        but the loaded matrix is&nbsp;<code>${_matrixData.matrix_id}</code>.<br>
        Go to the Dataset page, select the same countries, and recompute the matrix.
      </div></div>`;
      return;
    }

    /* Legacy result (no matrix_id stored) — show a soft warning banner but still render */
    if (!hasInfo) {
      const banner = document.createElement("div");
      banner.className = "info-note";
      banner.style.marginBottom = "0.75rem";
      banner.innerHTML = "<strong>Note:</strong> This is a legacy result with no matrix provenance record. The chart may be incorrect if the matrix has changed since this result was computed.";
      chartEl.innerHTML = "";
      chartEl.appendChild(banner);
      const inner = document.createElement("div");
      inner.id = chartId + "-inner";
      chartEl.appendChild(inner);
      if (_activeTab === "dendrogram") DendrogramChart.render(chartId + "-inner", _currentResult, _matrixData);
      else if (_activeTab === "heatmap") HeatmapChart.render(chartId + "-inner", _currentResult, _matrixData);
      else if (_activeTab === "clustermap") ClustermapChart.render(chartId + "-inner", _currentResult, _matrixData);
      return;
    }

    if (_activeTab === "dendrogram") {
      DendrogramChart.render(chartId, _currentResult, _matrixData);
    } else if (_activeTab === "heatmap") {
      HeatmapChart.render(chartId, _currentResult, _matrixData);
    } else if (_activeTab === "clustermap") {
      ClustermapChart.render(chartId, _currentResult, _matrixData);
    }
  }

  /* ── Compare page ─────────────────────────────────────────────────────── */
  async function renderCompare() {
    const fa = document.getElementById("cmp-a").value;
    const fb = document.getElementById("cmp-b").value;
    if (!fa || !fb || fa === fb) {
      document.getElementById("compare-body").style.display  = "none";
      document.getElementById("compare-empty").style.display = "";
      return;
    }
    try {
      const [a, b] = await Promise.all([API.getResult(fa), API.getResult(fb)]);
      document.getElementById("compare-body").style.display  = "";
      document.getElementById("compare-empty").style.display = "none";

      /* Metric comparison row */
      const row = document.getElementById("cmp-metric-row");
      const silA = a.silhouette_score !== null ? a.silhouette_score.toFixed(3) : "n/a";
      const silB = b.silhouette_score !== null ? b.silhouette_score.toFixed(3) : "n/a";
      const winner = a.silhouette_score !== null && b.silhouette_score !== null
        ? (a.silhouette_score > b.silhouette_score ? "A" : "B") : "—";

      row.innerHTML = `
        <div class="stat-card"><div class="stat-value" style="font-size:1.1rem">${_algoLabel(a)}</div><div class="stat-label">Algorithm A</div></div>
        <div class="stat-card"><div class="stat-value">${a.n_clusters}</div><div class="stat-label">k (A)</div></div>
        <div class="stat-card accent-green"><div class="stat-value">${silA}</div><div class="stat-label">Silhouette A</div></div>
        <div class="stat-card"><div class="stat-value" style="font-size:1.1rem">${_algoLabel(b)}</div><div class="stat-label">Algorithm B</div></div>
        <div class="stat-card"><div class="stat-value">${b.n_clusters}</div><div class="stat-label">k (B)</div></div>
        <div class="stat-card accent-purple"><div class="stat-value">${silB}</div><div class="stat-label">Silhouette B</div></div>
        <div class="stat-card"><div class="stat-value" style="color:#f59e0b">${winner}</div><div class="stat-label">Better silhouette</div></div>
      `;
      row.style.display = "";

      _renderCmpColumn("cmp-col-a", a, fa);
      _renderCmpColumn("cmp-col-b", b, fb);
    } catch (e) {
      toast("Compare error: " + e.message, "error");
    }
  }

  function _algoLabel(r) {
    if (r.algorithm === "hierarchical")
      return `HAC&nbsp;(${r.linkage || "avg"})`;
    return "K-Medoids";
  }

  function _renderCmpColumn(colId, result, filename) {
    const PALETTE = [
      "#4f46e5","#16a34a","#d97706","#dc2626","#7c3aed",
      "#2563eb","#db2777","#0d9488","#ea580c","#65a30d",
    ];
    const col = document.getElementById(colId);
    col.innerHTML = `<div class="cmp-header">${filename}</div>`;

    Object.keys(result.clusters).map(Number).sort((a, b) => a - b).forEach(cid => {
      const members = result.clusters[cid];
      const medoid  = result.medoids[cid];
      const c       = PALETTE[cid % PALETTE.length];
      const card    = document.createElement("div");
      card.className = "cluster-card";
      card.style.cssText = `border-left:3px solid ${c};margin-bottom:0.75rem`;
      card.innerHTML = `
        <div class="cluster-card-header">
          <span class="cluster-badge" style="background:${c}1a;color:${c}">
            C${cid} &bull; ${members.length}
          </span>
          <span class="cluster-medoid">&#9670;&nbsp;<strong>${cleanLabel(medoid)}</strong></span>
        </div>
        <div class="cluster-member-list" style="max-height:120px">
          ${members.slice().sort().map(n =>
            `<div class="cluster-member${n===medoid?" is-medoid":""}">${cleanLabel(n)}</div>`
          ).join("")}
        </div>
      `;
      col.appendChild(card);
    });
  }

  /* ── Toast ────────────────────────────────────────────────────────────── */
  let _toastTimer;
  function toast(msg, type = "info") {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.className   = `toast ${type}`;
    el.classList.remove("hidden");
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => el.classList.add("hidden"), 4500);
  }

  /* ── Public API ───────────────────────────────────────────────────────── */
  return {
    init, navigate,
    selectRegion, selectAlgo,
    computeMatrix, runClustering,
    loadResult, refreshResults,
    switchTab, renderCompare,
    toast,
  };
})();

document.addEventListener("DOMContentLoaded", () => App.init());
