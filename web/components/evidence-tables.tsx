"use client";
import { useEffect, useState } from "react";
import { Info, Table2 } from "lucide-react";
type Row = Record<string, unknown>;
const rows = (v: unknown): Row[] => (Array.isArray(v) ? v : []);
const obj = (v: unknown): Row => (v && typeof v === "object" ? (v as Row) : {});
const num = (v: unknown) =>
  typeof v === "number"
    ? v !== 0 && Math.abs(v) < 0.001
      ? v.toExponential(2)
      : v.toFixed(3)
    : "Unavailable";
const mean = (v: unknown) =>
  typeof v === "number" ? v : Number(obj(v).mean ?? 0);
function value(v: unknown, key: string) {
  if (v === null || v === undefined) return "Unavailable";
  if (typeof v === "object") {
    const i = obj(v);
    return `${num(i.mean)} [${num(i.low)}, ${num(i.high)}]`;
  }
  if (typeof v === "number")
    return [
      "position",
      "users",
      "sample_size",
      "impressions",
      "feedback",
      "ef_search",
      "requests",
      "concurrency",
    ].includes(key)
      ? v.toLocaleString()
      : num(v);
  return String(v);
}
function Table({
  data,
  columns,
}: {
  data: Row[];
  columns: [string, string][];
}) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {columns.map(([key, label]) => (
              <th key={key}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((r, i) => (
            <tr key={i}>
              {columns.map(([key], j) => (
                <td key={key} className={j === 0 ? "row-label" : "numeric"}>
                  {value(r[key], key)}
                  {key === "coverage" && (
                    <small className="coverage-note">
                      Observed distinct share; resampling repeats readers and
                      usually covers fewer books. Range is not a confidence
                      interval.
                    </small>
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {!data.length && (
        <p className="empty-table">
          No observations are available for this module.
        </p>
      )}
    </div>
  );
}
function Title({ title, caption }: { title: string; caption: string }) {
  return (
    <div className="section-title">
      <h2>{title}</h2>
      <p>{caption}</p>
    </div>
  );
}
export function EvidenceModules({ evidence: e }: { evidence: Row }) {
  const rerank = obj(e.reranking);
  const ann = obj(e.pgvector);
  return (
    <>
      <Title
        title="Paired comparisons"
        caption="Differences are left model minus right model. Confidence intervals use paired users; p values use sign-flip randomization and Benjamini–Hochberg correction."
      />
      <Table
        data={rows(e.comparisons)}
        columns={[
          ["left", "Left model"],
          ["right", "Right model"],
          ["difference", "Difference · 95% CI"],
          ["p", "p value"],
          ["q", "BH q value"],
        ]}
      />
      <Title
        title="Look beyond the average"
        caption="Cold readers, cold books, and long-tail items are reported separately. Different slice cohorts must not be compared as if they were the same users."
      />
      <Table
        data={rows(e.cold_slices)}
        columns={[
          ["slice", "Slice"],
          ["label", "Model"],
          ["users", "Readers"],
          ["ndcg", "NDCG @ 10 · 95% CI"],
        ]}
      />
      {e.cold_reader_analysis && (
        <div className="data-note">
          <Info size={18} />
          <p>
            {String(
              obj(e.cold_reader_analysis).detail ??
                obj(e.cold_reader_analysis).explanation ??
                "Slice composition changes the difficulty of the ranking task; see the report for diagnostic counts.",
            )}
          </p>
        </div>
      )}
      <Title
        title="What makes it onto the shelf"
        caption="Novelty, diversity, and calibration are user means with bootstrap intervals. Coverage is observed distinct catalog share; its bootstrap range describes resampling variability, not a population confidence interval."
      />
      <Table
        data={rows(e.metrics)}
        columns={[
          ["label", "Model"],
          ["coverage", "Coverage · resampling range"],
          ["long_tail_share", "Tail share · 95% CI"],
          ["diversity", "Diversity · 95% CI"],
          ["calibration", "Calibration · 95% CI"],
        ]}
      />
      {rows(rerank.rows).length > 0 && (
        <>
          <Title
            title="The cost of each re-ranking step"
            caption={String(rerank.detail)}
          />
          <Table
            data={rows(rerank.rows)}
            columns={[
              ["stage", "Serving stage"],
              ["ndcg", "NDCG @ 10 · 95% CI"],
              ["tag_diversity", "Tag diversity · 95% CI"],
              ["genre_divergence", "Genre divergence · 95% CI"],
              ["long_tail_share", "Tail share · 95% CI"],
            ]}
          />
        </>
      )}
      {ann.status === "measured_local_postgres" && rows(ann.rows).length > 0 && (
        <>
          <Title
            title="Approximate search, measured against exact search"
            caption={String(ann.detail)}
          />
          <p className="chart-caption">
            {Number(ann.items).toLocaleString()} books · {String(ann.queries)} queries
            {" "}· {String(ann.dimensions)} dimensions · pgvector {String(ann.pgvector_version)}.
            {" "}Recall compares the same top 200 neighbors with training items excluded.
          </p>
          <Table
            data={[
              { method: "Exact SQL", ef_search: "—", recall_at_200: "Reference", ...obj(ann.exact) },
              ...rows(ann.rows).map((row) => ({ method: "HNSW", ...row })),
            ]}
            columns={[
              ["method", "Search"],
              ["ef_search", "ef_search"],
              ["recall_at_200", "Recall @ 200 · 95% CI"],
              ["p50_ms", "p50 · ms"],
              ["p99_ms", "p99 · ms"],
            ]}
          />
        </>
      )}
      <HostedLatency />
    </>
  );
}
function HostedLatency() {
  const [latency, setLatency] = useState<Row | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/data/latency.json`, {
      signal: controller.signal,
      cache: "no-store",
    })
      .then(async (response) => response.ok ? obj(await response.json()) : null)
      .then((data) => {
        if (!data || data.status !== "measured" || typeof data.url !== "string") return;
        const endpoint = new URL(data.url);
        if (endpoint.protocol !== "https:" || ["localhost", "127.0.0.1", "[::1]"].includes(endpoint.hostname)) return;
        if (typeof data.requests !== "number" || !Number.isFinite(data.p50_ms) || !Number.isFinite(data.p99_ms)) return;
        setLatency(data);
      })
      .catch(() => { /* Optional measured artifact: omit the panel until published. */ });
    return () => controller.abort();
  }, []);
  if (!latency) return null;
  return (
    <>
      <Title title="Hosted API latency" caption={String(latency.operation)} />
      <Table
        data={[{ ...latency, label: "Persisted shelf", p50: latency.p50 ?? latency.p50_ms, p99: latency.p99 ?? latency.p99_ms }]}
        columns={[
          ["label", "Operation"],
          ["requests", "Timed requests"],
          ["concurrency", "Concurrency"],
          ["p50", "p50 · ms · resampling range"],
          ["p99", "p99 · ms · resampling range"],
        ]}
      />
      <p className="chart-caption">
        Measured {String(latency.measured_at)} at {String(latency.url)} after
        {" "}{String(latency.warmup_requests)} warm-up requests. Model: {String(latency.model_version)}.
        {" "}{String(latency.detail)}
      </p>
    </>
  );
}
export function ProtocolComparison({
  rows: data,
  kind,
}: {
  rows: Row[];
  kind: string;
}) {
  const [asTable, setAsTable] = useState(false);
  const max =
    Math.max(
      0.01,
      ...data.map((r) =>
        Math.max(
          Number(obj(r.full).high ?? mean(r.full)),
          Number(obj(r.shortcut).high ?? mean(r.shortcut)),
        ),
      ),
    ) * 1.1;
  return (
    <div className="chart-panel">
      <div className="chart-title">
        <b>
          {kind === "split"
            ? "Source order vs random split"
            : "Full catalog vs sampled candidates"}
        </b>
        <button className="text-button" onClick={() => setAsTable(!asTable)}>
          <Table2 size={14} />
          {asTable ? "Show chart" : "Show table"}
        </button>
      </div>
      {asTable ? (
        <Table
          data={data}
          columns={[
            ["label", "Model"],
            ["full", "Reference · 95% CI"],
            ["shortcut", "Shortcut · 95% CI"],
            ["delta", "Change · 95% interval"],
          ]}
        />
      ) : (
        <>
          <div className="comparison-legend">
            <span>
              <i />
              Reference protocol
            </span>
            <span>
              <i />
              Shortcut protocol
            </span>
          </div>
          <div className="paired-chart">
            {data.map((r, i) => (
              <div key={i}>
                <span>{String(r.label)}</span>
                <div className="paired-track">
                  <i
                    style={{ left: `${(mean(r.full) / max) * 100}%` }}
                    title={`Reference ${value(r.full, "full")}`}
                  />
                  <b
                    style={{ left: `${(mean(r.shortcut) / max) * 100}%` }}
                    title={`Shortcut ${value(r.shortcut, "shortcut")}`}
                  />
                  <span
                    style={{
                      left: `${(Math.min(mean(r.full), mean(r.shortcut)) / max) * 100}%`,
                      width: `${(Math.abs(mean(r.shortcut) - mean(r.full)) / max) * 100}%`,
                    }}
                  />
                </div>
                <span className="delta-label">+{num(mean(r.delta))}</span>
              </div>
            ))}
          </div>
          <div className="paired-axis">
            <span>0</span>
            <span>{num(max / 2)}</span>
            <span>{num(max)}</span>
          </div>
        </>
      )}
      <p className="chart-caption">
        {kind === "split"
          ? "Random splitting also changes the eligible reader cohort. This is protocol sensitivity, not a causal estimate of future leakage."
          : "One held-out positive plus uniformly sampled negatives. All rows use the same evaluation readers as the reference."}{" "}
        Intervals are available in the table.
      </p>
    </div>
  );
}
export function OpeEvidence({ ope }: { ope: Row }) {
  const simulator = obj(ope.simulator),
    obd = obj(ope.open_bandit),
    crosscheck = obj(ope.crosscheck);
  const [misspecified, setMisspecified] = useState(false),
    [sampleSize, setSampleSize] = useState("250");
  const simRows = rows(simulator.rows).filter(
    (row) =>
      row.reward_model ===
        (misspecified ? "misspecified constant" : "oracle") &&
      String(row.sample_size) === sampleSize,
  );
  const policies = [...new Set(rows(obd.rows).map((r) => String(r.policy)))];
  const [policy, setPolicy] = useState(
    policies.includes("Official prior BTS approximation")
      ? "Official prior BTS approximation"
      : (policies[0] ?? "Uniform sanity check"),
  );
  return (
    <>
      <div className="ope-controls">
        <label>
          Log sample size{" "}
          <select
            value={sampleSize}
            onChange={(e) => setSampleSize(e.target.value)}
          >
            {[
              ...new Set(
                rows(simulator.rows).map((r) => String(r.sample_size)),
              ),
            ].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          <input
            type="checkbox"
            checked={misspecified}
            onChange={(e) => setMisspecified(e.target.checked)}
          />
          Misspecified reward model
        </label>
        <span>
          {String(simulator.seeds)} seeds · exact truth {num(simulator.truth)}
        </span>
      </div>
      <Table
        data={simRows}
        columns={[
          ["estimator", "Estimator"],
          ["bias_interval", "Bias · 95% CI"],
          ["variance_interval", "Variance · 95% CI"],
          ["coverage_interval", "Interval coverage · 95% CI"],
        ]}
      />
      <div className="data-note">
        <Info size={18} />
        <div>
          <b>
            {misspecified
              ? "The reward model is deliberately wrong."
              : "Validation against known simulator truth."}
          </b>
          <p>
            {misspecified
              ? "Direct-method bias reflects the constant misspecified reward model. DR can correct the error under the known logging propensities."
              : "The simulator generates rewards from known probabilities. The target policy’s exact expected value is computed separately from the estimators."}
          </p>
        </div>
      </div>
      <Title
        title="A published-source cross-check"
        caption={String(crosscheck.detail ?? "No cross-check available.")}
      />
      <div className="stat-grid three">
        <div className="stat">
          <span className="overline">ESTIMATOR COMPARISONS</span>
          <strong>
            {String(crosscheck.estimator_comparisons ?? "Unavailable")}
          </strong>
          <p>Same seeded simulator fixtures</p>
        </div>
        <div className="stat">
          <span className="overline">ABSOLUTE TOLERANCE</span>
          <strong>{num(crosscheck.tolerance)}</strong>
          <p>Point-estimate arithmetic only</p>
        </div>
        <div className="stat">
          <span className="overline">SCOPE</span>
          <strong>Source methods</strong>
          <p>Not an installed-library certification</p>
        </div>
      </div>
      {rows(obd.rows).length > 0 && (
        <>
          <Title
            title="Real feedback, limited support"
            caption={String(obd.detail)}
          />
          <div className="ope-controls">
            <label>
              Target policy{" "}
              <select
                value={policy}
                onChange={(e) => setPolicy(e.target.value)}
              >
                {policies.map((v) => (
                  <option key={v}>{v}</option>
                ))}
              </select>
            </label>
            <span>
              {String(obd.evaluation_rows)} held-out rows · campaign{" "}
              {String(obd.campaign)}
            </span>
          </div>
          <Table
            data={rows(obd.rows).filter((r) => r.policy === policy)}
            columns={[
              ["campaign", "Campaign"],
              ["estimator", "Estimator"],
              ["position", "Position"],
              ["sample_size", "Rows"],
              ["estimate", "Estimate · 95% CI"],
              ["effective_sample_size", "Effective sample size"],
              ["bts_reference", "BTS on-policy reward · 95% CI"],
            ]}
          />
        </>
      )}
    </>
  );
}
