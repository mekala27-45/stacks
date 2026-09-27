import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix, save_npz
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from packages.api.catalog import Catalog
from packages.api.database import Feedback, Impression, SessionRecord
from packages.api.main import assign_arm, create_app, session_hash
from packages.rerank import calibration_divergence, diversity, mmr, rerank


@pytest.fixture
def bundle(tmp_path):
    items = [
        {
            "id": str(i),
            "title": f"Book {i}",
            "author": f"Author {i // 2}",
            "genre": "fiction" if i % 3 else "science",
            "tags": ["fiction" if i % 3 else "science", str(i % 4)],
            "year": 2000 + i,
            "popularity": 100 - i,
        }
        for i in range(1, 41)
    ]
    similarities = {
        item["id"]: [
            {"id": other["id"], "score": 1 / (1 + abs(int(item["id"]) - int(other["id"])))}
            for other in items
            if other != item
        ]
        for item in items
    }
    for name, content in {
        "catalog": items,
        "similarity": similarities,
        "readers": [{"id": "1", "history": ["1", "2", "3"]}],
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(content), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    positive = np.zeros((2, 40), dtype=np.float32)
    positive[0, :3] = 1
    positive[1, 3:] = 1
    matrix = csr_matrix(positive)
    seen = positive.copy()
    seen[0, 3] = 1
    rng = np.random.default_rng(52)
    np.savez_compressed(
        artifacts / "model-artifacts.npz",
        popularity=np.asarray(matrix.sum(axis=0)).ravel(),
        user_factors=rng.random((2, 3)).astype(np.float32),
        item_factors=rng.random((40, 3)).astype(np.float32),
        user_ids=np.array([1, 2]),
        catalog_ids=np.arange(1, 41),
    )
    save_npz(artifacts / "train-positive.npz", matrix)
    save_npz(artifacts / "train-seen.npz", csr_matrix(seen))
    cosine = rng.random((40, 40)).astype(np.float32)
    np.fill_diagonal(cosine, 0)
    save_npz(artifacts / "cosine.npz", csr_matrix(cosine))
    save_npz(artifacts / "content-vectors.npz", csr_matrix(rng.random((40, 3))))
    return tmp_path


@pytest.fixture
def service(bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "test-administrator")
    database_url = f"sqlite:///{tmp_path / 'test.sqlite'}"
    app = create_app(database_url, bundle, "test-persistent-hash-secret")
    with TestClient(app) as client:
        yield client, database_url
    app.state.engine.dispose()


def open_session(client, reader_id=None):
    response = client.post("/v1/session", json={"reader_id": reader_id})
    assert response.status_code == 201, response.text
    return response.json()


def test_durable_impressions_feedback_and_sessions_from_independent_connection(service):
    client, url = service
    shelf = open_session(client, "1")
    token = shelf["session_id"]
    assert len(shelf["items"]) == 10
    assert not {"1", "2", "3"} & {item["id"] for item in shelf["items"]}
    clicked = shelf["items"][0]
    response = client.post(
        f"/v1/session/{token}/event", json={"impression_id": clicked["impression_id"], "event": "click"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == 2
    assert clicked["id"] not in {item["id"] for item in response.json()["items"]}
    independent_engine = create_engine(url)
    with Session(independent_engine) as db:
        rows = list(db.scalars(select(Impression).order_by(Impression.position)))
        events = list(db.scalars(select(Feedback)))
        sessions = list(db.scalars(select(SessionRecord)))
        assert len(rows) == 20 and len(events) == 1 and len(sessions) == 1
        assert sessions[0].session_hash == session_hash(token, "test-persistent-hash-secret")
        assert token not in json.dumps(sessions[0].latest_shelf)
        assert clicked["id"] in sessions[0].history
        assert events[0].feedback_at >= db.get(Impression, events[0].impression_id).impression_at
        assert all(row.position > 0 and 0 < row.propensity <= 1 for row in rows)
        assert all(row.arm == shelf["arm"] and row.experiment_id == shelf["experiment_id"] for row in rows)
    independent_engine.dispose()


def test_probabilities_are_for_actual_conditional_policy(service):
    client, _ = service
    shelf = open_session(client)
    deterministic = shelf["items"][:-1]
    exploration = shelf["items"][-1]
    assert all(
        item["propensity"] == 1 and not item["trace"]["exploration"]["selected"] for item in deterministic
    )
    pool = exploration["trace"]["exploration"]["candidate_pool"]
    assert len(pool) == 20
    assert exploration["id"] in pool
    assert not set(pool) & {item["id"] for item in deterministic}
    assert exploration["propensity"] == pytest.approx(1 / len(pool))
    assert exploration["trace"]["shadow"]["served"] is False


def test_database_rejects_missing_propensity(service):
    client, url = service
    shelf = open_session(client)
    independent_engine = create_engine(url)
    with Session(independent_engine) as db:
        record = db.scalar(select(SessionRecord))
        db.add(
            Impression(
                id="invalid",
                session_hash=record.session_hash,
                item_id="4",
                position=1,
                propensity=None,
                arm=shelf["arm"],
                experiment_id="test",
                model_version="test",
                policy="deterministic",
                candidate_pool=["4"],
                trace={},
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
    independent_engine.dispose()


def test_assignment_and_trace_ownership(service):
    client, _ = service
    first, second = open_session(client), open_session(client)
    assignment = client.get("/v1/experiment/assign", params={"session_id": first["session_id"]}).json()
    assert assignment["arm"] == first["arm"]
    assert assign_arm("stable-session") == assign_arm("stable-session")
    url = f"/v1/explain/{first['items'][0]['impression_id']}"
    assert client.get(url, params={"session_id": first["session_id"]}).status_code == 200
    assert client.get(url, params={"session_id": second["session_id"]}).status_code == 404
    assert (
        client.post(
            f"/v1/session/{second['session_id']}/event",
            json={"impression_id": first["items"][0]["impression_id"], "event": "click"},
        ).status_code
        == 404
    )


def test_sse_replays_latest_revision_without_logging_more_impressions(service):
    client, url = service
    shelf = open_session(client)
    response = client.get(f"/v1/session/{shelf['session_id']}/stream?once=true")
    assert response.status_code == 200
    assert "event: shelf" in response.text and "id: 1" in response.text
    assert (
        client.get(f"/v1/session/{shelf['session_id']}/stream?once=true", headers={"Last-Event-ID": "1"}).text
        == ""
    )
    engine = create_engine(url)
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(Impression)) == 10
    engine.dispose()


def test_validation_duplicate_events_and_administrative_gate(service):
    client, _ = service
    shelf = open_session(client)
    token, impression = shelf["session_id"], shelf["items"][0]["impression_id"]
    event = {"impression_id": impression, "event": "click"}
    assert client.post("/v1/log", json={**event, "session_id": token}).status_code == 401
    assert client.post(f"/v1/session/{token}/event", json={**event, "rating": 5}).status_code == 422
    assert (
        client.post(
            f"/v1/session/{token}/event", json={"impression_id": impression, "event": "rating", "rating": 6}
        ).status_code
        == 422
    )
    assert client.post(f"/v1/session/{token}/event", json=event).status_code == 200
    assert client.post(f"/v1/session/{token}/event", json=event).status_code == 409
    assert client.get("/v1/recommend/new?limit=1000").status_code == 422
    assert client.get("/v1/similar/missing").status_code == 404
    assert client.post("/v1/session", json={"reader_id": "missing"}).status_code == 404


def test_empty_registry_evidence_is_persisted_as_rejection(service):
    client, _ = service
    response = client.post(
        "/v1/registry/evaluate",
        headers={"X-Admin-Token": "test-administrator"},
        json={"candidate_version": "challenger", "metrics": {}},
    )
    assert response.status_code == 201
    assert response.json()["eligible"] is False and response.json()["activated"] is False
    assert all(not gate["passed"] for gate in response.json()["gates"])


def test_write_payload_limit_and_registry_type_validation(service):
    client, _ = service
    assert client.post("/v1/session", content="x" * 32769).status_code == 413
    response = client.post(
        "/v1/registry/evaluate",
        headers={"X-Admin-Token": "test-administrator"},
        json={"candidate_version": "challenger", "metrics": {"coverage": True}},
    )
    assert response.status_code == 422


def test_reranking_contracts():
    candidates = [
        {
            "id": str(i),
            "author": str(i),
            "genre": "fiction" if i < 6 else "science",
            "tags": ["fiction" if i < 6 else "science"],
            "score": 1 - i / 100,
        }
        for i in range(12)
    ]
    assert mmr(candidates, 5, 0) == candidates[:5]
    assert diversity(mmr(candidates, 5)) >= diversity(candidates[:5])
    history = [candidates[8], candidates[9]]
    result, trace, _ = rerank(candidates, {"0"}, history, 5)
    assert "0" not in {item["id"] for item in result}
    assert trace["calibration"]["after"] <= trace["calibration"]["before"] + 1e-12
    assert calibration_divergence(history, history) == 0


def test_served_artifact_scores_match_shared_evaluation_scorer(bundle):
    source = Catalog(bundle)
    history = source.reader_history("1")
    expected = source.bundle.score(source.user_indices["1"])
    for model in ("als", "blend", "popularity", "item_cosine"):
        rows, mode = source.score(history, model, "1")
        assert mode == "evaluated-reader-exact"
        for row in rows:
            assert row["score"] == float(expected[model][source.indices[row["id"]]])
    rows, mode = source.score([*history, "5"], "blend", "1")
    assert mode == "frozen-item-factor-session-fold-in"
    assert set(source.reader_seen("1")) == {"1", "2", "3", "4"}


def test_resume_preferences_durable_logs_and_load_traffic_exclusion(service):
    client, _ = service
    shelf = client.post("/v1/session", json={"reader_id": "1", "traffic_kind": "load_test"}).json()
    token = shelf["session_id"]
    assert client.get(f"/v1/session/{token}").json()["revision"] == 1
    response = client.post(f"/v1/session/{token}/preferences", json={"genre": "fiction"})
    assert response.status_code == 200
    assert all(item["genre"] == "fiction" for item in response.json()["items"])
    logs = client.get(f"/v1/session/{token}/logs").json()
    assert len(logs["impressions"]) == 20
    assert all(
        "session_hash" not in row and row["trace"]["traffic_kind"] == "load_test"
        for row in logs["impressions"]
    )
    ope = client.get(f"/v1/session/{token}/ope").json()
    assert ope["load_test_excluded"] == 2 and ope["pending"] == 0 and ope["matured"] == 0
    assert client.get("/v1/ope").json()["load_test_excluded"] == 2
    assert client.post(f"/v1/session/{token}/preferences", json={"genre": "missing"}).status_code == 422


def test_owned_matured_exploration_logs_use_fixed_click_horizon(service):
    client, url = service
    shelf = open_session(client)
    token = shelf["session_id"]
    item = shelf["items"][-1]
    assert client.get(f"/v1/session/{token}/ope").json()["pending"] == 1
    assert (
        client.post(
            f"/v1/session/{token}/event", json={"impression_id": item["impression_id"], "event": "click"}
        ).status_code
        == 200
    )
    independent = create_engine(url)
    now = datetime.now(UTC)
    with Session(independent) as db:
        impression = db.get(Impression, item["impression_id"])
        impression.impression_at = now - timedelta(seconds=70)
        feedback = db.scalar(select(Feedback).where(Feedback.impression_id == impression.id))
        feedback.feedback_at = now - timedelta(seconds=20)
        db.commit()
    result = client.get(f"/v1/session/{token}/ope").json()
    assert result["status"] == "measured" and result["matured"] == 1 and result["pending"] == 1
    assert result["full_slate_identified"] is False
    target = item["trace"]["exploration"]["target_probabilities"][
        item["trace"]["exploration"]["candidate_pool"].index(item["id"])
    ]
    assert result["estimates"]["IPS"]["mean"] == pytest.approx(target / item["propensity"])
    assert result["estimates"]["SNIPS"]["mean"] == 1
    with Session(independent) as db:
        feedback = db.scalar(select(Feedback))
        feedback.feedback_at = now
        db.commit()
    late = client.get(f"/v1/session/{token}/ope").json()
    assert late["late_feedback_ignored"] == 1 and late["estimates"]["IPS"]["mean"] == 0
    monitoring = client.get("/v1/monitoring", headers={"X-Admin-Token": "test-administrator"}).json()
    assert monitoring["shadow"]["measured_shelves"] == 2
    assert len(monitoring["by_position"]) == 10
    independent.dispose()


def test_registry_check_uses_local_evidence_and_measured_latency(service, bundle):
    client, _ = service
    metrics = {
        "model": "als",
        "recall200": {"mean": 0.8},
        "coverage": {"mean": 0.5},
        "long_tail_share": {"mean": 0.4},
        "calibration": {"mean": 0.1},
    }
    (bundle / "artifacts" / "manifest.json").write_text(
        json.dumps(
            {
                "metrics": [metrics],
                "comparisons": [
                    {
                        "left": "als",
                        "right": "popularity",
                        "difference": {"low": 0.01, "high": 0.1},
                        "q": 0.01,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (bundle / "artifacts" / "latency.json").write_text(json.dumps({"p99_ms": 100}), encoding="utf-8")
    response = client.post("/v1/registry/check/als", headers={"X-Admin-Token": "test-administrator"})
    assert response.status_code == 201, response.text
    assert response.json()["eligible"] is True and response.json()["activated"] is False
    assert len(client.get("/v1/registry").json()["decisions"]) == 1


def test_out_of_process_api_session_stream_and_database_read(bundle, tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    database_url = f"sqlite:///{tmp_path / 'external.sqlite'}"
    environment = {
        **os.environ,
        "DATABASE_URL": database_url,
        "STACKS_DATA_DIR": str(bundle),
        "SESSION_HASH_KEY": "external-test-hash-key",
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "packages.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-access-log",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=3) as client:
            for _ in range(60):
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.ConnectError:
                    time.sleep(0.1)
            else:
                pytest.fail("External API did not start")
            shelf = client.post("/v1/session", json={}).json()
            token = shelf["session_id"]
            assert "event: shelf" in client.get(f"/v1/session/{token}/stream?once=true").text
            for _ in range(2):
                response = client.post(
                    f"/v1/session/{token}/event",
                    json={"impression_id": shelf["items"][0]["impression_id"], "event": "click"},
                )
                assert response.status_code == 200
                shelf = response.json()
            assert shelf["revision"] == 3
        script = "from sqlalchemy import create_engine,text; import os,json; e=create_engine(os.environ['DATABASE_URL']); c=e.connect(); print(json.dumps([c.execute(text('select count(*) from '+t)).scalar() for t in ['sessions','impressions','feedback']]))"
        observed = subprocess.check_output([sys.executable, "-c", script], env=environment, text=True)
        assert json.loads(observed) == [1, 30, 2]
    finally:
        process.terminate()
        process.wait(timeout=10)
