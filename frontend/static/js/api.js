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
    getMatrix:      (n)         => _req("GET",  n ? `/api/matrix?n=${n}` : "/api/matrix"),
    runCluster:     (body)      => _req("POST", "/api/cluster", body),
    listResults:    ()          => _req("GET",  "/api/results"),
    getResult:      (filename)  => _req("GET",  `/api/result/${filename}`),
  };
})();
