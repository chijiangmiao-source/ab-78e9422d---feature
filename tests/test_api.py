"""Tests for the HTTP-facing serialization layer (pure, no server needed)."""

from app.server import run_audit


K4 = {
    "nodes": ["A", "B", "C", "D"],
    "edges": [
        {"id": "e1", "u": "A", "v": "B", "length": 1},
        {"id": "e2", "u": "A", "v": "C", "length": 1},
        {"id": "e3", "u": "A", "v": "D", "length": 1},
        {"id": "e4", "u": "B", "v": "C", "length": 1},
        {"id": "e5", "u": "B", "v": "D", "length": 1},
        {"id": "e6", "u": "C", "v": "D", "length": 1},
    ],
    "start": "A",
}


def test_success_payload():
    r = run_audit(K4)
    assert r["ok"] is True
    assert r["optimalCount"] == 3
    assert r["addedLength"] == 2
    assert r["totalLength"] == 6
    assert r["canonicalVector"] == "001100"
    assert r["canonicalEdges"] == ["e3", "e4"]
    assert len(r["route"]) == 8
    # positions cover exactly duplicated canonical edges' extra copies
    counts = {e["id"]: len(r["positions"][str(e["index"])]) for e in r["edges"]}
    duplicated = set(r["canonicalEdges"])
    for eid, n in counts.items():
        assert n == (2 if eid in duplicated else 1)
    # route closes at start
    assert r["route"][0]["from"] == "A"
    assert r["route"][-1]["to"] == "A"


def test_eulerian_payload():
    payload = {
        "nodes": ["A", "B", "C"],
        "edges": [
            {"id": "a", "u": "A", "v": "B", "length": 3},
            {"id": "b", "u": "B", "v": "C", "length": 4},
            {"id": "c", "u": "C", "v": "A", "length": 5},
        ],
        "start": "B",
    }
    r = run_audit(payload)
    assert r["ok"]
    assert r["eulerian"] is True
    assert r["addedLength"] == 0
    assert r["optimalCount"] == 1
    assert r["canonicalVector"] == "000"
    assert r["canonicalEdges"] == []
    assert all(e["classification"] == "never" for e in r["edges"])


def test_failure_payload_locations():
    r = run_audit({"nodes": ["A", "B"],
                   "edges": [{"id": "x", "u": "A", "v": "Z", "length": 1}],
                   "start": "A"})
    assert r["ok"] is False
    assert "edges" in r["fields"]
    assert r["locations"][0]["row"] == 0

    r = run_audit({"nodes": ["A", "B"],
                   "edges": [{"id": "x", "u": "A", "v": "B", "length": -2}],
                   "start": "A"})
    assert not r["ok"] and "正整数" in r["error"]

    r = run_audit({"nodes": ["A", "B"],
                   "edges": [{"id": "x", "u": "A", "v": "B", "length": 1}],
                   "start": "Q"})
    assert not r["ok"] and r["fields"] == ["start"]

    r = run_audit({"nodes": ["A", "B", "C"],
                   "edges": [{"id": "x", "u": "A", "v": "B", "length": 1}],
                   "start": "A"})
    assert not r["ok"] and "不连通" in r["error"]


def test_malformed_payload_is_safe():
    assert run_audit({})["ok"] is False
    assert run_audit({"nodes": "ab", "edges": None})["ok"] is False


# ---------------------------------------------------------------------------
# Repeat-budget mode (热循环许可)
# ---------------------------------------------------------------------------

DIAMOND = {
    "nodes": ["A", "B", "C", "D"],
    "edges": [
        {"id": "ab", "u": "A", "v": "B", "length": 1},
        {"id": "bc", "u": "B", "v": "C", "length": 1},
        {"id": "ad", "u": "A", "v": "D", "length": 1},
        {"id": "dc", "u": "D", "v": "C", "length": 1},
        {"id": "ac", "u": "A", "v": "C", "length": 5},
    ],
    "start": "A",
}


def test_budget_payload_fields():
    r = run_audit({**DIAMOND, "maxRepeats": 1})
    assert r["ok"] is True
    assert r["repeatBudget"] == 1
    assert r["addedLength"] == 5  # longer but fewer duplicated segments
    assert r["addedLengthDelta"] == 3
    assert r["duplicatedCount"] == 1
    assert r["optimalCount"] == 1
    assert r["canonicalVector"] == "01000"
    assert r["canonicalEdges"] == ["ac"]
    assert len(r["route"]) == 6  # 5 original edges + 1 repeated copy
    assert r["route"][0]["from"] == "A" and r["route"][-1]["to"] == "A"


def test_budget_keys_absent_when_disabled():
    # mode off: response shape is exactly the pre-existing one
    r = run_audit(K4)
    assert r["ok"] is True
    assert "repeatBudget" not in r
    assert "duplicatedCount" not in r
    assert "addedLengthDelta" not in r
    r = run_audit({**K4, "maxRepeats": None})
    assert r["ok"] is True
    assert "repeatBudget" not in r


def test_budget_infeasible_payload():
    payload = {
        "nodes": ["A", "B", "C", "D"],
        "edges": [
            {"id": "e1", "u": "A", "v": "B", "length": 1},
            {"id": "e2", "u": "B", "v": "C", "length": 1},
            {"id": "e3", "u": "C", "v": "D", "length": 1},
        ],
        "start": "A",
        "maxRepeats": 2,
    }
    r = run_audit(payload)
    assert r["ok"] is False
    assert "预算不可行" in r["error"]
    assert "maxRepeats" in r["fields"]
    assert r["locations"][0]["field"] == "maxRepeats"


def test_budget_invalid_payload():
    for bad in (33, -1, "x", 1.5, True):
        r = run_audit({**K4, "maxRepeats": bad})
        assert r["ok"] is False
        assert "maxRepeats" in r["fields"]
        assert r["locations"][0]["field"] == "maxRepeats"
    r = run_audit({**K4, "maxRepeats": "2"})
    assert r["ok"] is True and r["repeatBudget"] == 2
