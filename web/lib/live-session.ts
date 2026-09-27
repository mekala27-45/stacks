"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { basePath, type Book, type Recommendation } from "./types";

type Row = Record<string, unknown>;
export type LiveShelf = {
  session_id?: string;
  revision: number;
  model_version: string;
  artifact_version?: string;
  scoring_mode?: string;
  arm: string;
  history?: (string | number)[];
  genre?: string;
  items: Row[];
};
export type LiveMode = "connecting" | "live" | "fallback";
const obj = (value: unknown): Row =>
  value && typeof value === "object" ? (value as Row) : {};

export function asRecommendation(
  row: Row,
  catalog: Book[],
  model: string,
): Recommendation {
  const id = Number(row.id ?? row.item_id);
  const book = catalog.find((item) => item.id === id);
  if (!book) throw new Error("The API model and book collection do not match.");
  const trace = obj(row.trace),
    ranking = obj(trace.ranking),
    features = obj(ranking.features);
  return {
    ...book,
    score: Number(row.score),
    position: Number(row.position),
    propensity: Number(row.propensity),
    explanation: String(row.explanation ?? "Evaluated model recommendation."),
    impression_id: String(row.impression_id),
    exploration: Boolean(obj(trace.exploration).selected),
    source: String(obj(trace.retrieval).backend ?? model),
    affinity: Number(features.affinity ?? 0),
    collaborative: Number(features.item_cosine ?? 0),
    trace,
    model_version: model,
  };
}

export function useLiveSession(
  catalog: Book[] | undefined,
  reader: string,
  enabled: boolean,
) {
  const [mode, setMode] = useState<LiveMode>("connecting");
  const [shelf, setShelf] = useState<Recommendation[]>([]);
  const [info, setInfo] = useState<LiveShelf | null>(null);
  const [ope, setOpe] = useState<Row | null>(null);
  const [monitoring, setMonitoring] = useState<Row | null>(null);
  const [reason, setReason] = useState("");
  const [generation, setGeneration] = useState(0);
  const connection = useRef<{
    base: string;
    token: string;
    stream: EventSource;
    accept: (shelf: LiveShelf) => void;
  } | null>(null);
  useEffect(() => {
    if (!catalog || !enabled) return;
    let active = true;
    let stream: EventSource | null = null;
    let disconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let reconnectNotice: ReturnType<typeof setTimeout> | undefined;
    let lastRevision = -1;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12000);
    setMode("connecting");
    setOpe(null);
    const accept = (next: LiveShelf) => {
      if (
        !active ||
        next.revision <= lastRevision ||
        !Array.isArray(next.items)
      )
        return;
      const items = next.items.map((row) =>
        asRecommendation(row, catalog, next.model_version),
      );
      lastRevision = next.revision;
      setShelf(items);
      setInfo(next);
      setMode("live");
      setReason("");
    };
    const start = async () => {
      const config = await fetch(`${basePath}/data/runtime.json`, {
        signal: controller.signal,
      }).then((r) => r.json());
      const base = String(
        process.env.NEXT_PUBLIC_API_URL ?? config.api_url ?? "",
      ).replace(/\/$/, "");
      if (!base) throw new Error("The hosted API is not configured.");
      let previous: { reader?: string; token?: string } = {};
      try {
        previous = JSON.parse(
          sessionStorage.getItem("stacks-live-session") ?? "{}",
        );
      } catch {
        /* Start a new session. */
      }
      let response: Response | undefined;
      if (previous.reader === reader && previous.token) {
        response = await fetch(
          `${base}/v1/session/${encodeURIComponent(previous.token)}`,
          { signal: controller.signal },
        );
      }
      if (!response?.ok) {
        response = await fetch(`${base}/v1/session`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            reader_id: reader === "visitor" ? null : reader,
          }),
          signal: controller.signal,
        });
      }
      if (!response.ok)
        throw new Error(`The API is unavailable (${response.status}).`);
      const data = (await response.json()) as LiveShelf;
      const token = data.session_id ?? previous.token;
      if (!token) throw new Error("The API did not return a session.");
      if (!active) return;
      accept(data);
      clearTimeout(timeout);
      sessionStorage.setItem(
        "stacks-live-session",
        JSON.stringify({ token, reader }),
      );
      stream = new EventSource(
        `${base}/v1/session/${encodeURIComponent(token)}/stream`,
      );
      connection.current = { base, token, stream, accept };
      stream.addEventListener("shelf", (event) => {
        try {
          accept(JSON.parse((event as MessageEvent).data));
        } catch {
          setReason("The stream returned an incompatible shelf.");
        }
      });
      stream.onerror = () => {
        // Short completed SSE responses reconnect normally on the free host.
        // Announce a disruption only if reconnection takes longer than usual.
        if (!reconnectNotice)
          reconnectNotice = setTimeout(() => {
            if (active)
              setReason("Reconnecting to the recommendation stream. Your last shelf is retained.");
          }, 4000);
        if (!disconnectTimer)
          disconnectTimer = setTimeout(() => {
            if (!active) return;
            stream?.close();
            connection.current = null;
            setMode("fallback");
            setReason(
              "The live stream disconnected. Reconnect to resume the persisted session.",
            );
          }, 15000);
      };
      stream.onopen = () => {
        clearTimeout(reconnectNotice);
        reconnectNotice = undefined;
        clearTimeout(disconnectTimer);
        disconnectTimer = undefined;
        if (active) setReason("");
      };
    };
    start()
      .catch((error: unknown) => {
        if (active) {
          setMode("fallback");
          setReason(
            error instanceof Error
              ? error.message
              : "The API could not be reached.",
          );
        }
      })
      .finally(() => clearTimeout(timeout));
    return () => {
      active = false;
      clearTimeout(timeout);
      clearTimeout(disconnectTimer);
      clearTimeout(reconnectNotice);
      controller.abort();
      stream?.close();
      connection.current = null;
    };
  }, [catalog, reader, enabled, generation]);

  const feedback = useCallback(
    async (impressionId: string, event: "click" | "save") => {
      const live = connection.current;
      if (!live)
        throw new Error("Your live session is reconnecting. Please try again.");
      const response = await fetch(
        `${live.base}/v1/session/${encodeURIComponent(live.token)}/event`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ impression_id: impressionId, event }),
          signal: AbortSignal.timeout(15000),
        },
      );
      if (!response.ok)
        throw new Error(
          `Your feedback was not saved (${response.status}). Please try again.`,
        );
      // The same revision also arrives over SSE; the revision guard deduplicates it.
      live.accept(await response.json());
    },
    [],
  );
  const loadOpe = useCallback(async () => {
    const live = connection.current;
    if (!live) return;
    try {
      const response = await fetch(
        `${live.base}/v1/session/${encodeURIComponent(live.token)}/ope`,
        { signal: AbortSignal.timeout(15000) },
      );
      if (response.ok) setOpe(await response.json());
    } catch {
      setReason("The latest observation summary could not be loaded.");
    }
  }, []);
  const loadMonitoring = useCallback(async () => {
    const live = connection.current;
    if (!live) return;
    try {
      const response = await fetch(`${live.base}/v1/monitoring`, {
        signal: AbortSignal.timeout(15000),
      });
      if (response.ok) setMonitoring(await response.json());
    } catch {
      setReason("The latest service observations could not be loaded.");
    }
  }, []);
  const preferences = useCallback(async (genre: string) => {
    const live = connection.current;
    if (!live) return;
    const response = await fetch(
      `${live.base}/v1/session/${encodeURIComponent(live.token)}/preferences`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ genre }),
        signal: AbortSignal.timeout(15000),
      },
    );
    if (!response.ok)
      throw new Error("Your genre preference could not be saved.");
    live.accept(await response.json());
  }, []);
  const logs = useCallback(async (): Promise<Row[]> => {
    const live = connection.current;
    if (!live) return [];
    const response = await fetch(
      `${live.base}/v1/session/${encodeURIComponent(live.token)}/logs`,
      { signal: AbortSignal.timeout(15000) },
    );
    if (!response.ok)
      throw new Error("The persisted logs could not be exported.");
    const data = await response.json();
    if (Array.isArray(data)) return data;
    const rows: Row[] = data.rows ?? data.impressions ?? [];
    const feedback: Row[] = data.feedback ?? [];
    return rows.map((row) => ({
      ...row,
      feedback: feedback.filter(
        (event) => event.impression_id === (row.id ?? row.impression_id),
      ),
    }));
  }, []);
  return {
    mode,
    shelf,
    info,
    ope,
    reason,
    feedback,
    preferences,
    loadOpe,
    monitoring,
    loadMonitoring,
    logs,
    retry: () => setGeneration((value) => value + 1),
  };
}
