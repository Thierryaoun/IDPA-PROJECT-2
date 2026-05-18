/**
 * api.js — Thin fetch wrapper for all Flask endpoints.
 */

const API = (() => {
  const BASE = "";   // same origin

  async function _req(method, path, body) {
    const opts = {
      method,
      headers: { "Content-Type": "application/json" },
    };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(BASE + path, opts);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  return {
    getCountries:   ()          => _req("GET",  "/api/countries"),
    getStatus:      ()          => _req("GET",  "/api/status"),
    startMatrix:    (body)      => _req("POST", "/api/matrix/start", body),
    pollJob:        (id)        => _req("GET",  `/api/job/${id}`),

    /* opts: { id: "matrix_id" } preferred, or { n: 25 } fallback, or omit for current */
    getMatrix: (opts) => {
      if (!opts) return _req("GET", "/api/matrix");
      if (typeof opts === "object") {
        if (opts.id) return _req("GET", `/api/matrix?id=${encodeURIComponent(opts.id)}`);
        if (opts.n)  return _req("GET", `/api/matrix?n=${opts.n}`);
      }
      if (typeof opts === "number") return _req("GET", `/api/matrix?n=${opts}`);
      return _req("GET", "/api/matrix");
    },

    /* Verify a result file matches a given matrix_id */
    validateMatrix: (resultFile, matrixId) =>
      _req("GET", `/api/validate_matrix?result=${encodeURIComponent(resultFile)}&matrix_id=${encodeURIComponent(matrixId)}`),

    runCluster:  (body)     => _req("POST", "/api/cluster", body),
    listResults: ()         => _req("GET",  "/api/results"),
    getResult:   (filename) => _req("GET",  `/api/result/${filename}`),
  };
})();
