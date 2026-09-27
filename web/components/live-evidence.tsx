"use client";
import { useEffect } from "react";
import type { useLiveSession } from "@/lib/live-session";

type Row = Record<string, unknown>;
const number = (value: unknown): string =>
  typeof value === "number" ? value.toFixed(4) : String(value ?? "Unavailable");
function estimate(value: unknown): string {
  if (!value || typeof value !== "object") return number(value);
  const row = value as Row;
  return `${number(row.mean ?? row.estimate ?? row.value)} [${number(row.low ?? row.lower)}, ${number(row.high ?? row.upper)}]`;
}

export function LiveEvidence({
  live,
}: {
  live: ReturnType<typeof useLiveSession>;
}) {
  const { mode, loadOpe, ope } = live;
  useEffect(() => {
    if (mode !== "live") return;
    void loadOpe();
    const timer = setInterval(() => void loadOpe(), 15000);
    return () => clearInterval(timer);
  }, [mode, loadOpe]);
  const rows = Array.isArray(ope?.rows)
    ? (ope.rows as Row[])
    : ope?.estimates && typeof ope.estimates === "object"
      ? (Object.entries(ope.estimates).map(([estimator, estimate]) => ({
          estimator,
          estimate,
        })) as Row[])
      : [];
  return (
    <section
      className="live-evidence"
      aria-label="Live session off-policy evaluation"
    >
      <div className="section-title">
        <h2>The live session</h2>
        <p>
          A fixed reward window and a supported final-slot target make this
          estimate interpretable.
        </p>
      </div>
      {mode !== "live" ? (
        <div className="data-note">
          <p>
            The model service is{" "}
            {mode === "connecting" ? "connecting" : "unavailable"}. Static
            browser logs are not mixed with persisted API observations.
          </p>
        </div>
      ) : (
        <>
          <div className="data-note">
            <div>
              <b>Conditional final-slot evaluation</b>
              <p>
                {String(
                  ope?.detail ??
                    ope?.estimand ??
                    "The target chooses the highest-scored book 80% of the time and explores uniformly 20%, within each logged final-slot candidate pool. A click within 60 seconds is the binary reward. Deterministic slots and unsupported catalog actions are excluded.",
                )}
              </p>
            </div>
          </div>
          <div className="stat-grid three">
            <div className="stat">
              <span className="overline">MATURED OBSERVATIONS</span>
              <strong>
                {number(ope?.matured ?? ope?.matured_rows ?? ope?.n)}
              </strong>
              <p>Completed reward windows</p>
            </div>
            <div className="stat">
              <span className="overline">PENDING</span>
              <strong>{number(ope?.pending ?? ope?.pending_rows)}</strong>
              <p>Not counted as zero rewards early</p>
            </div>
            <div className="stat">
              <span className="overline">STATUS</span>
              <strong>{String(ope?.status ?? "Loading")}</strong>
              <p>
                {String(
                  ope?.interval_method ??
                    "Intervals account for bounded importance weights",
                )}
              </p>
            </div>
          </div>
          {rows.length > 0 && (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Estimator</th>
                    <th>Estimate · interval</th>
                    <th>Effective sample size</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, index) => (
                    <tr key={index}>
                      <td>{String(row.estimator)}</td>
                      <td className="numeric">
                        {estimate(row.estimate ?? row)}
                      </td>
                      <td className="numeric">
                        {number(
                          row.effective_sample_size ??
                            ope?.effective_sample_size,
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="subtle">
            Small samples can produce very wide bounds. This estimates the
            defined final-slot reward in demonstration sessions; it does not
            establish full-slate value or production lift.
          </p>
          <button className="button outline" onClick={() => void loadOpe()}>
            Refresh matured observations
          </button>
        </>
      )}
    </section>
  );
}
