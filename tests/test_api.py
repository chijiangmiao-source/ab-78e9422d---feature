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
# Repeat-segment cap mode
# ---------------------------------------------------------------------------


THIN = {
    "nodes": ["A", "X", "Y", "D"],
    "edges": [
        {"id": "d1", "u": "A", "v": "D", "length": 10},
        {"id": "d2", "u": "A", "v": "D", "length": 100},
        {"id": "p1", "u": "A", "v": "X", "length": 1},
        {"id": "p2", "u": "X", "v": "Y", "length": 1},
        {"id": "p3", "u": "Y", "v": "D", "length": 1},
    ],
    "start": "A",
}


def test_budget_mode_payload():
    payload = dict(THIN, repeatCap=1)
    r = run_audit(payload)
    assert r["ok"] is True
    assert r["budgetFeasible"] is True
    assert r["repeatCap"] == 1
    assert r["repeatCount"] == 1
    assert r["addedLength"] == 10
    assert r["unconstrainedAddedLength"] == 3
    assert r["addedLengthDelta"] == 7
    assert r["canonicalVector"] == "10000"
    assert r["canonicalEdges"] == ["d1"]
    # copies and route stay consistent with the duplicate set
    counts = {e["id"]: len(r["positions"][str(e["index"])]) for e in r["edges"]}
    assert counts["d1"] == 2 and all(counts[k] == 1 for k in ("d2", "p1", "p2", "p3"))
    assert r["route"][0]["from"] == "A" and r["route"][-1]["to"] == "A"


def test_budget_mode_infeasible_payload():
    path = {
        "nodes": ["A", "B", "C", "D"],
        "edges": [
            {"id": "e1", "u": "A", "v": "B", "length": 1},
            {"id": "e2", "u": "B", "v": "C", "length": 1},
            {"id": "e3", "u": "C", "v": "D", "length": 1},
        ],
        "start": "A",
        "repeatCap": 2,
    }
    r = run_audit(path)
    assert r["ok"] is True  # input is legal; only the budget fails
    assert r["budgetFeasible"] is False
    assert r["repeatCap"] == 2
    assert r["minimumRepeatCount"] == 3
    assert r["unconstrainedAddedLength"] == 3
    assert "route" not in r


def test_budget_mode_eulerian_zero_cap():
    payload = {
        "nodes": ["A", "B", "C"],
        "edges": [
            {"id": "a", "u": "A", "v": "B", "length": 3},
            {"id": "b", "u": "B", "v": "C", "length": 4},
            {"id": "c", "u": "C", "v": "A", "length": 5},
        ],
        "start": "B",
        "repeatCap": 0,
    }
    r = run_audit(payload)
    assert r["ok"] and r["budgetFeasible"] is True
    assert r["addedLength"] == 0 and r["repeatCount"] == 0
    assert r["addedLengthDelta"] == 0


def test_budget_mode_omitted_leaves_response_unchanged():
    r = run_audit(K4)
    assert "repeatCap" not in r
    assert "budgetFeasible" not in r
    assert "addedLengthDelta" not in r


def test_budget_mode_invalid_cap_is_validation_error():
    r = run_audit(dict(K4, repeatCap=33))
    assert r["ok"] is False
    assert "repeatCap" in r["fields"]
    assert r["locations"] and r["locations"][0]["field"] == "repeatCap"
    r = run_audit(dict(K4, repeatCap=-1))
    assert r["ok"] is False and "repeatCap" in r["fields"]
    r = run_audit(dict(K4, repeatCap="x"))
    assert r["ok"] is False and "repeatCap" in r["fields"]
