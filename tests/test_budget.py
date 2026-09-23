"""Tests for repeat-budget mode (热循环许可): T-joins of bounded size.

The oracle is a brute-force scan over all edge subsets keeping only those
that are parity-closed (odd-degree set == the graph's odd vertices) and use
at most ``cap`` distinct edges; the minimum weight over that feasible
domain, the exact co-optimal sets, the canonical 0-preferred vector, the
three-way classification, the added-length delta against the unconstrained
optimum and the closed route are all cross-checked against the solver's
meet-in-the-middle computation.
"""

import random

import pytest

from app.solver import AuditError, audit
from tests.test_solver import check_route, edge, random_graph


def brute_budget(nodes, raw_edges, cap):
    """(best, [frozenset], odd) with |S| <= cap; best is None if infeasible."""
    m = len(raw_edges)
    deg = {n: 0 for n in nodes}
    for e in raw_edges:
        deg[e["u"]] += 1
        deg[e["v"]] += 1
    odd = frozenset(n for n in nodes if deg[n] % 2)

    best = None
    sets = []
    for mask in range(1 << m):
        if mask.bit_count() > cap:
            continue
        d = dict.fromkeys(nodes, 0)
        w = 0
        for i in range(m):
            if mask >> i & 1:
                e = raw_edges[i]
                d[e["u"]] += 1
                d[e["v"]] += 1
                w += e["length"]
        if frozenset(n for n in nodes if d[n] % 2) != odd:
            continue
        if best is None or w < best:
            best = w
            sets = [frozenset(i for i in range(m) if mask >> i & 1)]
        elif w == best:
            sets.append(frozenset(i for i in range(m) if mask >> i & 1))
    return best, sets, odd


# Diamond: two 2-length paths A-B-C / A-D-C and a direct edge A-C of 5.
# Odd vertices are A and C.  Edges sorted by id: ab, ac, ad, bc, dc.
DIAMOND_NODES = list("ABCD")
DIAMOND_EDGES = [
    edge("ab", "A", "B", 1),
    edge("bc", "B", "C", 1),
    edge("ad", "A", "D", 1),
    edge("dc", "D", "C", 1),
    edge("ac", "A", "C", 5),
]


# ---------------------------------------------------------------------------
# Deterministic cases
# ---------------------------------------------------------------------------


def test_budget_forces_longer_fewer_segments_solution():
    # cap 1: the unconstrained optima {ab,bc} / {ad,dc} (added 2, size 2)
    # are infeasible; the longer direct edge {ac} (added 5, size 1) wins.
    r = audit(DIAMOND_NODES, DIAMOND_EDGES, "A", 1)
    assert r.repeat_budget == 1
    assert r.added_length == 5
    assert r.optimal_count == 1
    assert r.canonical_set == frozenset({1})  # ac
    assert r.bit_vector == "01000"
    assert r.duplicated_count == 1
    assert r.added_length_delta == 3  # unconstrained optimum is 2
    assert r.classification[1] == "required"
    assert all(r.classification[i] == "never" for i in (0, 2, 3, 4))
    check_route(r, "A")


def test_budget_counts_equal_weight_different_size_optima():
    # Three equal-weight A-B channels: direct edge b0 (w2, size 1) and two
    # 2-edge paths (w2, size 2).  All three sets are co-optimal at cap 2;
    # the size-1 set shares its half's parity mask with a size-2 set at the
    # same weight, and both must be counted.
    nodes = list("ABXY")
    raw = [
        edge("a1", "A", "Y", 1),
        edge("a2", "Y", "B", 1),
        edge("b0", "A", "B", 2),
        edge("b1", "A", "X", 1),
        edge("b2", "X", "B", 1),
    ]
    r = audit(nodes, raw, "A", 2)
    assert r.added_length == 2
    assert r.optimal_count == 3  # {b0}, {b1,b2}, {a1,a2}
    assert r.bit_vector == "00011"
    assert r.duplicated_count == 2
    assert r.added_length_delta == 0
    check_route(r, "A")
    r = audit(nodes, raw, "A", 1)
    assert r.optimal_count == 1 and r.bit_vector == "00100"
    assert r.duplicated_count == 1
    check_route(r, "A")


def test_budget_loose_enough_recovers_unconstrained_result():
    r = audit(DIAMOND_NODES, DIAMOND_EDGES, "A", 2)
    assert r.added_length == 2
    assert r.optimal_count == 2
    assert r.added_length_delta == 0
    assert r.bit_vector == "00101"  # {ad,dc} beats {ab,bc} at bit 0
    assert r.duplicated_count == 2
    check_route(r, "A")
    base = audit(DIAMOND_NODES, DIAMOND_EDGES, "A")
    assert (r.added_length, r.optimal_count, r.bit_vector) == (
        base.added_length,
        base.optimal_count,
        base.bit_vector,
    )
    assert r.classification == base.classification
    assert base.repeat_budget is None  # mode off: no budget fields


def test_budget_infeasible_path_graph():
    nodes = list("ABCD")
    raw = [edge("e1", "A", "B", 1), edge("e2", "B", "C", 1), edge("e3", "C", "D", 1)]
    # the only T-join is the whole path (3 segments); cap 2 cannot close
    with pytest.raises(AuditError) as ei:
        audit(nodes, raw, "A", 2)
    assert "预算不可行" in ei.value.message
    assert "maxRepeats" in ei.value.fields
    assert ei.value.locations[0]["field"] == "maxRepeats"
    r = audit(nodes, raw, "A", 3)
    assert r.added_length == 3 and r.optimal_count == 1
    assert r.duplicated_count == 3
    check_route(r, "A")


def test_budget_zero_boundaries():
    # Eulerian network: empty set always fits, zero augmentation
    nodes = list("ABC")
    raw = [edge("a", "A", "B", 3), edge("b", "B", "C", 4), edge("c", "C", "A", 5)]
    r = audit(nodes, raw, "B", 0)
    assert r.added_length == 0 and r.optimal_count == 1
    assert r.bit_vector == "000"
    assert r.canonical_set == frozenset()
    assert r.duplicated_count == 0
    assert r.added_length_delta == 0
    assert all(v == "never" for v in r.classification.values())
    check_route(r, "B")
    # non-Eulerian network with cap 0: no closed walk exists
    with pytest.raises(AuditError) as ei:
        audit(list("AB"), [edge("e", "A", "B", 1)], "A", 0)
    assert "预算不可行" in ei.value.message


def test_budget_exact_count_parallel_chain():
    # 4 segments in series, 7 parallel unit edges each: P0-P1-P2-P3-P4.
    # Only P0 and P4 are odd; every T-join uses an odd count per segment,
    # so the minimum is one edge per segment: 7**4 distinct optimal sets.
    nodes = ["P0", "P1", "P2", "P3", "P4"]
    raw = []
    for seg in range(4):
        for j in range(7):
            raw.append(edge(f"s{seg}e{j}", f"P{seg}", f"P{seg + 1}", 1))
    r = audit(nodes, raw, "P0", 4)
    assert r.added_length == 4
    assert r.optimal_count == 7**4  # 2401, exact big-integer count
    assert r.duplicated_count == 4
    assert r.added_length_delta == 0
    assert all(v == "optional" for v in r.classification.values())
    check_route(r, "P0")
    # cap 3 < 4 segments needed -> infeasible
    with pytest.raises(AuditError):
        audit(nodes, raw, "P0", 3)


# ---------------------------------------------------------------------------
# Budget validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, 33, 1.5, "x", "", "  ", True, [2], {"v": 2}])
def test_budget_validation_rejects_bad_values(bad):
    with pytest.raises(AuditError) as ei:
        audit(list("AB"), [edge("e", "A", "B", 1)], "A", bad)
    assert "maxRepeats" in ei.value.fields
    assert ei.value.locations[0]["field"] == "maxRepeats"


def test_budget_accepts_int_and_numeric_string():
    r = audit(list("AB"), [edge("e", "A", "B", 2)], "A", 1)
    assert r.repeat_budget == 1 and r.added_length == 2
    r = audit(list("AB"), [edge("e", "A", "B", 2)], "A", "1")
    assert r.repeat_budget == 1 and r.added_length == 2


# ---------------------------------------------------------------------------
# Randomized cross-validation against the brute-force oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_budget_brute_force_crosscheck(seed):
    rng = random.Random(10_000 + seed)
    nnodes = rng.randint(2, 7)
    nedges = rng.randint(nnodes - 1, min(13, nnodes * (nnodes - 1) // 2))
    nodes, raw = random_graph(rng, nnodes, nedges)
    start = rng.choice(nodes)
    cap = rng.choice([0, 1, 2, 3, 4, 5, 17, 32])

    best, sets, odd = brute_budget(nodes, raw, cap)
    if best is None:
        with pytest.raises(AuditError) as ei:
            audit(nodes, raw, start, cap)
        assert "预算不可行" in ei.value.message
        return

    r = audit(nodes, raw, start, cap)
    ids_sorted = [e.eid for e in r.edges]
    raw_id = {i: e["id"] for i, e in enumerate(raw)}
    id_sets = [{raw_id[i] for i in s} for s in sets]

    assert set(r.odd_vertices) == odd
    assert r.added_length == best
    assert r.optimal_count == len(sets)
    canon_ids = {r.edges[i].eid for i in r.canonical_set}
    assert canon_ids in id_sets

    in_all = set(raw_id.values())
    in_any: set = set()
    for s in id_sets:
        in_all &= s
        in_any |= s
    for i, eid in enumerate(ids_sorted):
        want = (
            "required" if eid in in_all
            else "optional" if eid in in_any
            else "never"
        )
        assert r.classification[i] == want

    def vec(idset):
        return "".join("1" if eid in idset else "0" for eid in ids_sorted)

    assert vec(canon_ids) == min(vec(s) for s in id_sets)

    # budget-mode extras
    assert r.repeat_budget == cap
    assert r.duplicated_count == len(r.canonical_set)
    unbest, _, _ = brute_budget(nodes, raw, 32)  # 32 >= m: unconstrained
    assert r.added_length_delta == r.added_length - unbest
    check_route(r, start)


def test_budget_largest_bounds_run_fast():
    # 18 nodes, 32 edges, budget mode must stay fast (meet in the middle).
    rng = random.Random(42)
    nodes, raw = random_graph(rng, 18, 32)
    r = audit(nodes, raw, nodes[0], 17)
    check_route(r, nodes[0])
    assert len(r.route) == 32 + len(r.canonical_set)
