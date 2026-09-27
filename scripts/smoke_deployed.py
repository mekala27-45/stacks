"""Verify a deployed model API with excluded test traffic and a genuine live SSE update."""

import argparse
import json
import math
import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import httpx

ORIGIN = "https://mekala27-45.github.io"
TARGET = "final-slot-80-best-20-uniform-v1"
Record = dict[str, object]


class SmokeFailure(Exception):
    """A safe diagnostic that never includes a response body or session URL."""


def record(value: object) -> Record:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SmokeFailure("Expected a JSON object")
    return cast(Record, value)


def records(value: object) -> list[Record]:
    if not isinstance(value, list):
        raise SmokeFailure("Expected a JSON array")
    return [record(row) for row in value]


def text(value: object) -> str:
    if not isinstance(value, str):
        raise SmokeFailure("Expected a string field")
    return value


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SmokeFailure("Expected a finite numeric field")
    return float(value)


def decoded(response: httpx.Response, expected: int = 200) -> Record:
    if response.status_code != expected:
        raise SmokeFailure(f"Unexpected HTTP status {response.status_code}; expected {expected}")
    return record(cast(object, response.json()))


def check(checks: dict[str, bool], name: str, condition: bool) -> None:
    checks[name] = condition
    if not condition:
        raise SmokeFailure(f"Check failed: {name}")


def next_shelf(lines: Iterator[str]) -> tuple[str, Record]:
    event, identifier = "", ""
    data: list[str] = []
    for line in lines:
        if not line:
            if event == "shelf" and data:
                return identifier, record(cast(object, json.loads("\n".join(data))))
            event, identifier, data = "", "", []
        elif line.startswith("event:"):
            event = line[6:].lstrip()
        elif line.startswith("id:"):
            identifier = line[3:].lstrip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    raise SmokeFailure("SSE closed before the expected shelf event")


def verify_shelf(shelf: Record, checks: dict[str, bool], label: str, artifact: str) -> list[Record]:
    items = records(shelf.get("items"))
    check(checks, f"{label}_ten_unique_items", len(items) == 10 and len({text(i.get("id")) for i in items}) == 10)
    check(checks, f"{label}_versioned_model", shelf.get("schema_version") == "1.1"
          and shelf.get("artifact_version") == artifact
          and shelf.get("arm") in ("als", "blend")
          and shelf.get("model_version") == f"{shelf.get('arm')}:{artifact}")
    check(checks, f"{label}_test_traffic", shelf.get("traffic_kind") == "load_test")
    check(checks, f"{label}_ordered_positions", [item.get("position") for item in items] == list(range(1, 11)))
    for position, item in enumerate(items):
        trace = record(item.get("trace"))
        exploration = record(trace.get("exploration"))
        pool = exploration.get("candidate_pool")
        if not isinstance(pool, list) or not pool or any(not isinstance(book, str) for book in pool):
            raise SmokeFailure("Invalid exploration candidate pool")
        pool_ids = cast(list[str], pool)
        probability = number(item.get("propensity"))
        is_final = position == 9
        check(checks, f"{label}_position_{position + 1}_propensity",
              trace.get("traffic_kind") == "load_test"
              and item.get("id") in pool_ids and len(set(pool_ids)) == len(pool_ids)
              and (len(pool_ids) == 20 if is_final else pool_ids == [item.get("id")])
              and math.isclose(probability, 1 / len(pool_ids), abs_tol=1e-12)
              and probability == number(exploration.get("propensity"))
              and exploration.get("policy") == ("uniform" if is_final else "deterministic")
              and exploration.get("selected") is is_final
              and exploration.get("target_policy") == TARGET
              and exploration.get("reward_horizon_seconds") == 60)
    return items


def verify(url: str, report: Record, checks: dict[str, bool]) -> None:
    timeout = httpx.Timeout(60.0, read=20.0)
    headers = {"Origin": ORIGIN}
    with (
        httpx.Client(base_url=url, timeout=timeout, headers=headers) as creator,
        httpx.Client(base_url=url, timeout=timeout, headers=headers) as stream_client,
        httpx.Client(base_url=url, timeout=timeout, headers=headers) as observer,
    ):
        health_response = creator.get("/health")
        health = decoded(health_response)
        check(checks, "public_health", health.get("status") == "ok" and health.get("backend") == "live-edge-d1")
        check(checks, "health_exact_cors", health_response.headers.get("Access-Control-Allow-Origin") == ORIGIN)
        preflight = creator.options("/v1/session", headers={"Access-Control-Request-Method": "POST",
                                                           "Access-Control-Request-Headers": "Content-Type"})
        check(checks, "preflight_exact_cors", preflight.status_code == 204
              and preflight.headers.get("Access-Control-Allow-Origin") == ORIGIN
              and "POST" in preflight.headers.get("Access-Control-Allow-Methods", "")
              and "content-type" in preflight.headers.get("Access-Control-Allow-Headers", "").lower())
        denied = creator.options("/v1/session", headers={"Origin": "https://unrelated.example"})
        check(checks, "cors_other_origin_not_granted", "Access-Control-Allow-Origin" not in denied.headers)
        artifact = text(health.get("artifact_version"))
        # Only safe, validated identifiers enter the report; the session token stays in local variables.
        check(checks, "artifact_identifier", re.fullmatch(r"[A-Za-z0-9._-]{1,100}", artifact) is not None)
        report["artifact_version"] = artifact
        created_response = creator.post("/v1/session", json={"reader_id": "2", "traffic_kind": "load_test"})
        shelf = decoded(created_response, 201)
        check(checks, "write_exact_cors", created_response.headers.get("Access-Control-Allow-Origin") == ORIGIN)
        items = verify_shelf(shelf, checks, "initial", artifact)
        check(checks, "initial_evaluated_reader", shelf.get("scoring_mode") == "evaluated-reader-exact")
        check(checks, "initial_revision", shelf.get("revision") == 1)
        token = text(shelf.get("session_id"))
        if re.fullmatch(r"[a-f0-9]{64}", token) is None:
            raise SmokeFailure("Invalid session token format")
        session_path = f"/v1/session/{token}"
        selected = items[-1]
        impression = text(selected.get("impression_id"))
        check(checks, "impression_identifier", re.fullmatch(r"[a-f0-9-]{36}", impression) is not None)
        report["selected_impression_id"] = impression
        report["model_version"] = text(shelf.get("model_version"))
        started = time.perf_counter()
        with stream_client.stream("GET", f"{session_path}/stream", headers={"Accept": "text/event-stream"}) as stream:
            check(checks, "real_sse_response", stream.status_code == 200
                  and stream.headers.get("Content-Type", "").startswith("text/event-stream")
                  and stream.headers.get("Access-Control-Allow-Origin") == ORIGIN)
            lines = stream.iter_lines()
            event_id, first = next_shelf(lines)
            first_event_ms = (time.perf_counter() - started) * 1000
            check(checks, "sse_initial_delivery_under_five_seconds", first_event_ms <= 5000)
            report["initial_event_ms"] = first_event_ms
            check(checks, "sse_initial_revision", event_id == "1" and first.get("revision") == 1
                  and first.get("items") == shelf.get("items"))
        updated = decoded(observer.post(f"{session_path}/event",
                                        json={"impression_id": impression, "event": "click"}))
        verify_shelf(updated, checks, "updated", artifact)
        check(checks, "event_revision_and_foldin", updated.get("revision") == 2
              and updated.get("scoring_mode") == "frozen-item-factor-session-fold-in")
        with stream_client.stream("GET", f"{session_path}/stream",
                                  headers={"Accept": "text/event-stream", "Last-Event-ID": event_id}) as stream:
            check(checks, "sse_reconnect_response", stream.status_code == 200
                  and stream.headers.get("Content-Type", "").startswith("text/event-stream"))
            event_id, pushed = next_shelf(stream.iter_lines())
            check(checks, "sse_reconnected_revision", event_id == "2" and pushed.get("revision") == 2
                  and pushed.get("items") == updated.get("items"))
        report["transport"] = "bounded_sse_reconnect"
        report["transport_detail"] = (
            "The managed proxy buffers event-stream chunks until response closure. The Worker uses two-second "
            "connections and a one-second EventSource retry with Last-Event-ID. This verification receives "
            "the initial event, clicks through an independent client, then receives the new revision on a "
            "resumed SSE connection; it does not claim continuously flushed chunks on one connection."
        )
        resumed = decoded(observer.get(session_path))
        check(checks, "independent_resume", resumed.get("revision") == 2
              and resumed.get("items") == updated.get("items"))
        logs = decoded(observer.get(f"{session_path}/logs"))
        impressions, feedback = records(logs.get("impressions")), records(logs.get("feedback"))
        check(checks, "persisted_twenty_impressions", len(impressions) == 20)
        check(checks, "persisted_one_owned_feedback", len(feedback) == 1
              and feedback[0].get("impression_id") == impression and feedback[0].get("event") == "click")
        check(checks, "persisted_test_traffic", all(record(row.get("trace")).get("traffic_kind") == "load_test"
                                                    for row in impressions))
        check(checks, "logs_no_identity_fields", all("session_hash" not in row and "session_id" not in row
                                                       for row in impressions + feedback))
        check(checks, "logs_match_returned_exposures", {row.get("id") for row in impressions}
              == {row.get("impression_id") for row in items + records(updated.get("items"))})
        for row in impressions:
            trace = record(record(row.get("trace")).get("exploration"))
            check(checks, "persisted_propensities_match", row.get("propensity") == trace.get("propensity"))
        selected_log = next(row for row in impressions if row.get("id") == impression)
        exposure_time = datetime.fromisoformat(text(selected_log.get("impression_at")))
        feedback_time = datetime.fromisoformat(text(feedback[0].get("feedback_at")))
        check(checks, "separate_exposure_feedback_times", 0 <= (feedback_time - exposure_time).total_seconds() <= 60)
        ope = decoded(observer.get(f"{session_path}/ope"))
        check(checks, "scripted_traffic_excluded_from_ope", ope.get("load_test_excluded") == 2
              and ope.get("matured") == 0 and ope.get("pending") == 0)
        report["counters"] = {"sse_events": 2, "sse_connections": 2, "revision": 2, "impressions": len(impressions),
                              "feedback": len(feedback), "ope_test_rows_excluded": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/deployment-verification.json"))
    args = parser.parse_args()
    url = str(args.url).rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query \
            or parsed.fragment or parsed.path:
        parser.error("Use the public HTTPS origin without credentials, path, query or fragment")
    checks: dict[str, bool] = {}
    report: Record = {"status": "running", "url": url, "started_at": datetime.now(UTC).isoformat(),
                      "traffic_kind": "load_test", "cors_origin": ORIGIN, "reader_id": "2", "checks": checks}
    try:
        verify(url, report, checks)
        report["status"] = "passed"
    except (SmokeFailure, httpx.HTTPError, ValueError, TypeError, KeyError, StopIteration) as error:
        report["status"] = "failed"
        # httpx exception strings contain request URLs, so never serialize them.
        report["error"] = str(error) if isinstance(error, SmokeFailure) else type(error).__name__
    report["finished_at"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks_passed": sum(checks.values()),
                      "checks_total": len(checks), "output": str(args.output)}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
