"""Chinese postman / route inspection core.

Undirected multigraph (parallel edges allowed, self loops forbidden) with
positive integer edge lengths.  Finds the minimum added length needed for all
vertices to have even degree, the exact number of optimal duplicate sets, the
canonical duplicate set (0-preferred bit vector in edge order), per-edge
classification, and a closed Euler tour for the canonical augmentation.

Exact counting without enumeration
----------------------------------
A duplicate set is a T-join: in the subgraph formed by the duplicated edges
exactly the originally odd vertices T have odd degree.  T-join theorem:

  * minimum T-join weight = minimum weight of a perfect matching of T under
    the shortest-path metric;
  * every minimum T-join decomposes into edge-disjoint shortest paths whose
    endpoint pairs form such a minimum matching.

For each odd pair (i, j) let A[i][j] be the number of shortest i-j paths
(parallel edges count separately).  Choices for the pairs of a matching are
independent -- two shortest paths of pairs inside one minimum matching cannot
share an edge, because their edge-union would then be a strictly cheaper
T-join.  Hence the number of optimal sets for a matching M is the product of
A[i][j] over its pairs, and different matchings give different sets.  The
total count is obtained by a weighted subset DP without ever enumerating the
sets:

    C[S] = sum over min-cost partners j of the first vertex:
               A[i][j] * C[S \\ {i, j}]

For per-edge classification, let B_e[i][j] be the number of shortest i-j
paths that use edge e (forward/backward shortest-path-count product through
the edge).  A second DP G_e[S] counts optimal sets for subproblem S that
contain e, using B_e for the pair whose path carries e and A - B_e otherwise:

    G_e[S] = sum over min-cost partners j:
               B_e[i][j] * C[S'] + (A[i][j] - B_e[i][j]) * G_e[S']

Edge e is required when G_e[T] == C[T], optional for 0 < G_e[T] < C[T], and
never duplicated when G_e[T] == 0.

The canonical set is built greedily in edge order: bit p is 0 whenever an
optimum T-join still exists with the edges pinned so far, the newly forced
edges toggling the parity target.
"""

from __future__ import annotations

import heapq
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

INF = 10**30
TOKEN_RE = re.compile(r"^[!-~]+$")  # printable non-space ASCII
REPEAT_CAP_MAX = 32


class AuditError(ValueError):
    """Validation failure with page locations."""

    def __init__(
        self,
        message: str,
        fields: Sequence[str] = (),
        locations: Sequence[dict] = (),
    ):
        super().__init__(message)
        self.message = message
        self.fields = tuple(fields)
        self.locations = list(locations)


class BudgetInfeasible(Exception):
    """Legal repeat-segment cap under which no parity-closing tour exists.

    Not an :class:`AuditError`: the input is valid, so the API keeps
    ``ok:true`` and reports the infeasibility separately.
    """

    def __init__(self, cap: int, min_required: int, unconstrained_added: int):
        super().__init__(f"重复段上限 {cap} 不可行")
        self.cap = cap
        self.min_required = min_required
        self.unconstrained_added = unconstrained_added


@dataclass(frozen=True)
class Edge:
    eid: str
    u: str
    v: str
    length: int
    index: int


@dataclass(frozen=True)
class RouteStep:
    edge_index: int
    edge_id: str
    frm: str
    to: str
    length: int
    duplicate_no: int  # which copy of this edge, 1-based, in traversal order


@dataclass
class AuditResult:
    nodes: List[str]
    edges: List[Edge]
    start: str
    odd_vertices: Tuple[str, ...]
    components: Tuple[Tuple[str, ...], ...]
    total_length: int
    added_length: int
    optimal_count: int
    canonical_set: FrozenSet[int]
    bit_vector: str
    classification: Dict[int, str]  # required | optional | never
    multiplicity: Tuple[int, ...]
    route: Tuple[RouteStep, ...]
    # repeat-segment budget mode (None / absent when the mode is disabled)
    repeat_cap: Optional[int] = None
    repeat_count: int = 0
    unconstrained_added_length: Optional[int] = None

    @property
    def is_eulerian(self) -> bool:
        return not self.odd_vertices

    @property
    def budget_enabled(self) -> bool:
        return self.repeat_cap is not None

    @property
    def added_length_delta(self) -> Optional[int]:
        if self.repeat_cap is None or self.unconstrained_added_length is None:
            return None
        return self.added_length - self.unconstrained_added_length


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_positive_int(value, eid: str) -> int:
    if isinstance(value, bool):
        raise AuditError(f"管段 {eid} 长度必须为正整数", ("edges",))
    if isinstance(value, int):
        length = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        length = int(value.strip())
    else:
        raise AuditError(f"管段 {eid} 长度必须为正整数", ("edges",))
    if length <= 0:
        raise AuditError(f"管段 {eid} 长度必须为正整数", ("edges",))
    return length


def validate_input(
    nodes: Sequence[str], raw_edges: Sequence[dict], start: Optional[str]
) -> Tuple[List[str], List[Edge]]:
    clean_nodes: List[str] = []
    node_rows: Dict[str, int] = {}
    for i, raw in enumerate(nodes):
        name = ("" if raw is None else str(raw)).strip()
        if name == "":
            continue
        loc = [{"field": "nodes", "row": i}]
        if not TOKEN_RE.match(name):
            raise AuditError(
                f"节点 {name!r} 必须为非空白 ASCII 字符", ("nodes",), loc
            )
        if name in node_rows:
            raise AuditError(
                f"节点 {name!r} 重复",
                ("nodes",),
                loc + [{"field": "nodes", "row": node_rows[name]}],
            )
        node_rows[name] = i
        clean_nodes.append(name)

    if not (2 <= len(clean_nodes) <= 18):
        raise AuditError(
            f"唯一节点数量为 {len(clean_nodes)}，必须在 2 至 18 之间",
            ("nodes",),
            [{"field": "nodes"}],
        )

    if not raw_edges:
        raise AuditError("至少需要 1 条管段", ("edges",), [{"field": "edges"}])
    if len(raw_edges) > 32:
        raise AuditError(
            f"管段数量为 {len(raw_edges)}，不能超过 32",
            ("edges",),
            [{"field": "edges"}],
        )

    edges: List[Edge] = []
    id_rows: Dict[str, int] = {}
    for i, re_ in enumerate(raw_edges):
        loc = [{"field": "edges", "row": i}]
        eid = str(re_.get("id", "") or "").strip()
        if not eid:
            raise AuditError(
                f"第 {i + 1} 条管段缺少唯一标识", ("edges",), loc
            )
        if not TOKEN_RE.match(eid):
            raise AuditError(
                f"管段标识 {eid!r} 必须为非空白 ASCII 字符", ("edges",), loc
            )
        if eid in id_rows:
            raise AuditError(
                f"管段标识 {eid!r} 重复",
                ("edges",),
                loc + [{"field": "edges", "row": id_rows[eid]}],
            )
        id_rows[eid] = i

        u = str(re_.get("u", "") or "").strip()
        v = str(re_.get("v", "") or "").strip()
        if u not in node_rows:
            raise AuditError(
                f"管段 {eid} 的端点 {u or '(空)'} 不是已声明节点",
                ("edges",),
                loc,
            )
        if v not in node_rows:
            raise AuditError(
                f"管段 {eid} 的端点 {v or '(空)'} 不是已声明节点",
                ("edges",),
                loc,
            )
        if u == v:
            raise AuditError(
                f"管段 {eid} 为自环（{u}），禁止自环", ("edges",), loc
            )

        length = _as_positive_int(re_.get("length", None), eid)
        edges.append(Edge(eid=eid, u=u, v=v, length=length, index=i))

    # The canonical bit vector is ordered by edge *identifier*, so reorder
    # the edges (and their indices) lexicographically now that validation of
    # rows/locations is done.
    edges.sort(key=lambda e: e.eid)
    edges = [
        Edge(eid=e.eid, u=e.u, v=e.v, length=e.length, index=i)
        for i, e in enumerate(edges)
    ]

    start_s = "" if start is None else str(start).strip()
    if not start_s:
        raise AuditError("请选择检修口", ("start",), [{"field": "start"}])
    if start_s not in node_rows:
        raise AuditError(
            f"检修口 {start_s!r} 不存在", ("start",), [{"field": "start"}]
        )

    return clean_nodes, edges


def parse_repeat_cap(value) -> Optional[int]:
    """Normalize the optional repeat-segment cap (0..REPEAT_CAP_MAX).

    Absent / falsey values disable the mode.  A present but invalid value is
    an :class:`AuditError` located at the budget control.
    """
    if value is None or isinstance(value, bool):
        if value is None:
            return None
        raise AuditError(
            f"重复段上限必须为 0 至 {REPEAT_CAP_MAX} 的整数",
            ("repeatCap",),
            [{"field": "repeatCap"}],
        )
    if isinstance(value, int):
        cap = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        cap = int(value.strip())
    elif isinstance(value, float) and value.is_integer():
        cap = int(value)
    else:
        raise AuditError(
            f"重复段上限必须为 0 至 {REPEAT_CAP_MAX} 的整数",
            ("repeatCap",),
            [{"field": "repeatCap"}],
        )
    if not (0 <= cap <= REPEAT_CAP_MAX):
        raise AuditError(
            f"重复段上限必须在 0 至 {REPEAT_CAP_MAX} 之间，当前为 {cap}",
            ("repeatCap",),
            [{"field": "repeatCap"}],
        )
    return cap


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def build_nadj(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    forbidden: FrozenSet[int] = frozenset(),
):
    nadj: Dict[str, List[Tuple[str, int, int]]] = {n: [] for n in nodes}
    for e in edges:
        if e.index in forbidden:
            continue
        nadj[e.u].append((e.v, e.index, e.length))
        nadj[e.v].append((e.u, e.index, e.length))
    return nadj


def adjacency(nodes: Sequence[str], edges: Sequence[Edge]):
    adj: Dict[str, List[Tuple[str, int]]] = {n: [] for n in nodes}
    for e in edges:
        adj[e.u].append((e.v, e.index))
        adj[e.v].append((e.u, e.index))
    return adj


def connected_components(nodes: Sequence[str], adj) -> List[List[str]]:
    comps: List[List[str]] = []
    unvisited = set(nodes)
    while unvisited:
        seed = next(iter(unvisited))
        stack = [seed]
        unvisited.discard(seed)
        comp: List[str] = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y, _ in adj[x]:
                if y in unvisited:
                    unvisited.discard(y)
                    stack.append(y)
        comps.append(sorted(comp))
    return comps


def dijkstra(nodes: Sequence[str], nadj, src: str):
    """Shortest distances from src."""
    dist = {src: 0}
    pq = [(0, src)]
    while pq:
        du, u = heapq.heappop(pq)
        if du != dist.get(u):
            continue
        for w, _, length in nadj[u]:
            nd = du + length
            if w not in dist or nd < dist[w]:
                dist[w] = nd
                heapq.heappush(pq, (nd, w))
    return dist


def shortest_path_masks(
    nadj,
    src: str,
    dst: str,
    dist: dict,
) -> Tuple[int, ...]:
    """All shortest src->dst paths as integer edge-index masks.

    DFS over the shortest-path DAG; positive edge lengths make it acyclic
    (every predecessor has strictly smaller distance).
    """
    if src == dst:
        return (0,)
    pred: Dict[str, List[Tuple[str, int]]] = {}
    for x, dx in dist.items():
        if x == src:
            continue
        ps = []
        for y, ei, length in nadj[x]:
            if y in dist and dist[y] + length == dx:
                ps.append((y, ei))
        pred[x] = ps

    result: List[int] = []
    cur = 0
    seen_v = {dst}

    def dfs(x: str):
        nonlocal cur
        if x == src:
            result.append(cur)
            return
        for y, ei in pred.get(x, ()):
            if y in seen_v:
                continue
            seen_v.add(y)
            cur |= 1 << ei
            dfs(y)
            cur &= ~(1 << ei)
            seen_v.discard(y)

    dfs(dst)
    return tuple(result)


# ---------------------------------------------------------------------------
# Perfect matching DP over a subset of vertices
# ---------------------------------------------------------------------------


def matching_dp(dist_matrix, labels: Sequence[int]):
    """Minimum matching cost and number of matchings attaining it.

    dist_matrix[i][j] is the shortest-path distance; INF means unreachable.
    Returns (cost array by mask, count array by mask).
    """
    k = len(labels)
    size = 1 << k
    cost = [INF] * size
    count = [0] * size
    cost[0] = 0
    count[0] = 1
    for mask in range(1, size):
        if mask.bit_count() & 1:
            continue
        i = (mask & -mask).bit_length() - 1
        rest0 = mask ^ (1 << i)
        best = INF
        total = 0
        bits = rest0
        while bits:
            jb = bits & -bits
            j = jb.bit_length() - 1
            bits ^= jb
            d = dist_matrix[i][j]
            sub = rest0 ^ jb
            if d >= INF or cost[sub] >= INF:
                continue
            val = d + cost[sub]
            if val < best:
                best = val
                total = count[sub]
            elif val == best:
                total += count[sub]
        cost[mask] = best
        count[mask] = total
    return cost, count


# ---------------------------------------------------------------------------
# Euler circuit (Hierholzer) on the expanded multigraph
# ---------------------------------------------------------------------------


def euler_circuit(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    start: str,
    multiplicity: Sequence[int],
) -> List[RouteStep]:
    copies: List[Tuple[int, str, str, int]] = []
    for e in edges:
        for _ in range(multiplicity[e.index]):
            copies.append((e.index, e.u, e.v, e.length))

    adj: Dict[str, List[int]] = {n: [] for n in nodes}
    for ci, (ei, u, v, _) in enumerate(copies):
        adj[u].append(ci)
        adj[v].append(ci)

    used = [False] * len(copies)
    stack: List[Tuple[str, int]] = [(start, -1)]
    circuit: List[Tuple[str, int]] = []
    while stack:
        x, _ = stack[-1]
        chosen: Optional[int] = None
        for ci in adj[x]:  # incident lists in ascending copy order
            if not used[ci]:
                chosen = ci
                break
        if chosen is None:
            circuit.append(stack.pop())
        else:
            used[chosen] = True
            _, u, v, _ = copies[chosen]
            stack.append((v if x == u else u, chosen))

    circuit.reverse()
    walk_copies = [ci for _, ci in circuit[1:]]

    steps: List[RouteStep] = []
    dup_counter: Dict[int, int] = {}
    cur = start
    for ci in walk_copies:
        ei, u, v, length = copies[ci]
        frm, to = (u, v) if cur == u else (v, u)
        dup_counter[ei] = dup_counter.get(ei, 0) + 1
        steps.append(
            RouteStep(
                edge_index=ei,
                edge_id=edges[ei].eid,
                frm=frm,
                to=to,
                length=length,
                duplicate_no=dup_counter[ei],
            )
        )
        cur = to
    return steps


# ---------------------------------------------------------------------------
# Enumeration of the distinct optimal T-join edge sets
# ---------------------------------------------------------------------------


def enumerate_optimal_tjoins(
    odd: Tuple[str, ...],
    dist_from: Dict[str, dict],
    nadj,
    costdp,
    dist_matrix,
):
    """All distinct minimum T-join masks for the odd vertices.

    dp[mask] is the set of distinct edge masks of optimal T-joins pairing
    exactly the odd vertices in ``mask``.  Anchor the lowest-index vertex i
    and combine a shortest i-j path with an optimal solution of the remaining
    mask via symmetric difference; integer masks deduplicate automatically.
    The matching cost DP gates which partners j can occur in an optimum.
    """
    k = len(odd)
    size = 1 << k
    pair_cache: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    def paths_between(a: int, b: int) -> Tuple[int, ...]:
        key = (a, b)
        if key not in pair_cache:
            pair_cache[key] = shortest_path_masks(
                nadj, odd[a], odd[b], dist_from[odd[a]]
            )
        return pair_cache[key]

    dp: List[Optional[set]] = [None] * size
    dp[0] = {0}
    for mask in range(1, size):
        if mask.bit_count() & 1:
            continue
        ib = mask & -mask
        i = ib.bit_length() - 1
        rest0 = mask ^ ib
        target_cost = costdp[mask]
        result: set = set()
        bits = rest0
        while bits:
            jb = bits & -bits
            j = jb.bit_length() - 1
            bits ^= jb
            sub = rest0 ^ jb
            if dist_matrix[i][j] >= INF or costdp[sub] >= INF:
                continue
            if dist_matrix[i][j] + costdp[sub] != target_cost:
                continue
            sub_sets = dp[sub]
            for pmask in paths_between(i, j):
                for base in sub_sets:
                    result.add(pmask ^ base)
        dp[mask] = result
    return dp[size - 1]


# ---------------------------------------------------------------------------
# Cardinality-constrained exact solver (repeat-segment budget mode)
# ---------------------------------------------------------------------------
#
# When a cap on the number of distinct duplicated edges is imposed, the
# cheapest feasible set need not be a minimum T-join of shortest paths: a
# longer but thinner set (fewer distinct edges) may be the only one fitting
# the budget.  The matching/shortest-path machinery therefore no longer
# applies; the problem is
#
#     minimize   sum_e w_e x_e
#     subject to {e : x_e = 1} has odd-degree set exactly T,
#                |{e : x_e = 1}| <= cap,
#
# i.e. exact minimization over edge subsets.  With m <= 32 a meet-in-the
#-middle split (two halves of <= 16 edges) enumerates 2*2^16 = 131072 subsets;
# halves meet on the GF(2) parity mask of incident vertices.  B-side subsets
# are aggregated by (parity, cardinality, weight), so the exact optimum, its
# co-optimal set count (arbitrary-precision Python ints) and per-edge
# inclusion counts are obtained without enumerating joined solutions.


@dataclass
class ConstrainedOptimum:
    added: int
    count: int
    canonical_mask: int
    in_all: int
    in_any: int


def _half_arrays(edge_list: Sequence[int], node_bit: Dict[str, int], edges):
    """Subset parity/weight/cardinality arrays for one MITM half."""
    h = len(edge_list)
    size = 1 << h
    xs = [0] * size
    ws = [0] * size
    cs = [0] * size
    for mask in range(1, size):
        lb = mask & -mask
        j = lb.bit_length() - 1
        prev = mask ^ lb
        e = edges[edge_list[j]]
        xs[mask] = xs[prev] ^ node_bit[e.u] ^ node_bit[e.v]
        ws[mask] = ws[prev] + e.length
        cs[mask] = cs[prev] + 1
    return edge_list, xs, ws, cs


def solve_cardinality_constrained(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    odd: Tuple[str, ...],
    cap: int,
) -> Optional[ConstrainedOptimum]:
    """Exact optimum among parity-closing sets with at most ``cap`` edges."""
    m = len(edges)
    node_bit = {n: 1 << i for i, n in enumerate(nodes)}
    target = 0
    for n in odd:
        target |= node_bit[n]

    split = m // 2
    a_edges = list(range(split))
    b_edges = list(range(split, m))
    _, xpa, wa, ca = _half_arrays(a_edges, node_bit, edges)
    _, xpb, wb, cb = _half_arrays(b_edges, node_bit, edges)
    size_a, size_b = len(xpa), len(xpb)

    # ---- aggregate B side: by parity, then weight -> counts per card ------
    # minCard[p] / minW[p][c] for the optimum-weight pass;
    # tab[p][w] = length-(hb+1) array, cumulative count with card <= c.
    min_card_b: Dict[int, int] = {}
    min_w_b: Dict[int, Dict[int, int]] = {}
    tab: Dict[int, Dict[int, list]] = {}
    for b in range(size_b):
        p, w, c = xpb[b], wb[b], cb[b]
        if p not in min_card_b or c < min_card_b[p]:
            min_card_b[p] = c
        d = min_w_b.setdefault(p, {})
        if c not in d or w < d[c]:
            d[c] = w
        wt = tab.setdefault(p, {})
        arr = wt.get(w)
        if arr is None:
            arr = [0] * (len(b_edges) + 1)
            wt[w] = arr
        arr[c] += 1
    # prefix minima / cumulative counts
    min_w_pref: Dict[int, List[int]] = {}
    hb = len(b_edges)
    for p, d in min_w_b.items():
        pref = [INF] * (hb + 1)
        run = INF
        for c in range(hb + 1):
            if c in d:
                run = min(run, d[c])
            pref[c] = run
        min_w_pref[p] = pref
    for wt in tab.values():
        for arr in wt.values():
            for c in range(1, hb + 1):
                arr[c] += arr[c - 1]

    # ---- pass 1: optimum weight (and feasible-cardinality gating) --------
    best = INF
    for a in range(size_a):
        q = target ^ xpa[a]
        cmax = cap - ca[a]
        if cmax < 0:
            continue
        mc = min_card_b.get(q)
        if mc is None or mc > cmax:
            continue
        pref = min_w_pref[q]
        val = wa[a] + (pref[cmax] if cmax <= hb else pref[hb])
        if val < best:
            best = val
    if best >= INF:
        return None

    # ---- pass 2: counts (big integers) -----------------------------------
    # f[a] = number of optimal B subsets joining A subset a under the cap.
    # Per-B-edge inclusion counts are accumulated via packed counters: each
    # B edge owns a SLOT-bit lane of one big integer, so adding a subset adds
    # one to every lane whose edge it contains -- one big-int add replaces a
    # full table build per edge.  Lane width 33 is provably enough: at most
    # 2^hb B subsets join at most 2^ha A subsets, product <= 2^32 < 2^33.
    SLOT = 33
    LANE_MASK = (1 << SLOT) - 1
    packtab: Dict[int, Dict[int, list]] = {}
    pack_inc = [0] * size_b  # packed unit contribution of each B subset
    for b in range(1, size_b):
        lb = b & -b
        j = lb.bit_length() - 1
        pack_inc[b] = pack_inc[b ^ lb] + (1 << (SLOT * j))
    for b in range(size_b):
        wt = packtab.setdefault(xpb[b], {})
        arr = wt.get(wb[b])
        if arr is None:
            arr = [0] * (hb + 1)
            wt[wb[b]] = arr
        arr[cb[b]] += pack_inc[b]
    for wt in packtab.values():
        for arr in wt.values():
            for c in range(1, hb + 1):
                arr[c] += arr[c - 1]

    f = [0] * size_a
    total = 0
    b_inclusion_packed = 0
    for a in range(size_a):
        cmax = cap - ca[a]
        if cmax < 0:
            continue
        q = target ^ xpa[a]
        wt = tab.get(q)
        if wt is None:
            continue
        cnt_arr = wt.get(best - wa[a])
        if cnt_arr is None:
            continue
        cclip = min(cmax, hb)
        n_b = cnt_arr[cclip]
        f[a] = n_b
        total += n_b
        b_inclusion_packed += packtab[q][best - wa[a]][cclip]

    # A-edge inclusion: superset zeta sums of f -> F[i] = sum_{a superset i}
    ha = len(a_edges)
    fz = f[:]
    for i in range(ha):
        bit = 1 << i
        for mask in range(size_a):
            if not (mask & bit):
                fz[mask] += fz[mask | bit]

    in_all = (1 << m) - 1
    in_any = 0
    for local_i in range(ha):
        ge = fz[1 << local_i]
        ei = a_edges[local_i]
        if ge:
            in_any |= 1 << ei
        if ge != total:
            in_all &= ~(1 << ei)
    for local_j in range(hb):
        ge = (b_inclusion_packed >> (SLOT * local_j)) & LANE_MASK
        ei = b_edges[local_j]
        if ge:
            in_any |= 1 << ei
        if ge != total:
            in_all &= ~(1 << ei)

    # ---- canonical mask: smallest 0-preferred vector among co-optima -----
    # A-side edges precede B-side edges in edge order, so choose the smallest
    # feasible A mask first (local bit 0 = MSB), then the smallest B mask.
    def rev_table(bits: int) -> List[int]:
        size = 1 << bits
        rev = [0] * size
        for x in range(1, size):
            rev[x] = (rev[x >> 1] >> 1) | ((x & 1) << (bits - 1))
        return rev

    reva = rev_table(len(a_edges))
    revb = rev_table(len(b_edges))
    best_a = None
    for a in range(size_a):
        cmax = cap - ca[a]
        if cmax < 0:
            continue
        wt = tab.get(target ^ xpa[a])
        if wt is None:
            continue
        arr = wt.get(best - wa[a])
        if arr is not None and arr[min(cmax, hb)] > 0:
            if best_a is None or reva[a] < reva[best_a]:
                best_a = a
    assert best_a is not None
    a = best_a
    cmax = cap - ca[a]
    q = target ^ xpa[a]
    wb_want = best - wa[a]
    best_b = None
    for b in range(size_b):
        if xpb[b] == q and wb[b] == wb_want and cb[b] <= cmax:
            if best_b is None or revb[b] < revb[best_b]:
                best_b = b
    assert best_b is not None
    canonical_mask = a | (best_b << split)

    return ConstrainedOptimum(
        added=best,
        count=total,
        canonical_mask=canonical_mask,
        in_all=in_all,
        in_any=in_any,
    )


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def _build_result(
    *,
    nodes,
    edges,
    start,
    odd,
    comps,
    total_length,
    added_length,
    optimal_count,
    canonical_mask,
    in_all,
    in_any,
    m,
    repeat_cap=None,
    repeat_count=None,
    unconstrained_added=None,
) -> AuditResult:
    """Assemble the result dataclass from an optimum described by masks."""
    bit_vector = "".join(
        "1" if canonical_mask >> i & 1 else "0" for i in range(m)
    )
    classification: Dict[int, str] = {}
    for i in range(m):
        if in_all >> i & 1:
            classification[i] = "required"
        elif in_any >> i & 1:
            classification[i] = "optional"
        else:
            classification[i] = "never"

    multiplicity = tuple(
        1 + (1 if canonical_mask >> i & 1 else 0) for i in range(m)
    )
    route = euler_circuit(nodes, edges, start, multiplicity)
    if repeat_count is None:
        repeat_count = canonical_mask.bit_count()

    kwargs = {}
    if repeat_cap is not None:
        kwargs = dict(
            repeat_cap=repeat_cap,
            repeat_count=repeat_count,
            unconstrained_added_length=unconstrained_added,
        )
    return AuditResult(
        nodes=nodes,
        edges=edges,
        start=start,
        odd_vertices=odd,
        components=tuple(tuple(c) for c in comps),
        total_length=total_length,
        added_length=int(added_length),
        optimal_count=optimal_count,
        canonical_set=frozenset(
            i for i in range(m) if canonical_mask >> i & 1
        ),
        bit_vector=bit_vector,
        classification=classification,
        multiplicity=multiplicity,
        route=tuple(route),
        **kwargs,
    )


def _unconstrained_added(
    nodes: Sequence[str], edges: Sequence[Edge], odd: Tuple[str, ...]
) -> int:
    """Minimum T-join weight under the original (unbounded) problem."""
    nadj = build_nadj(nodes, edges)
    dist_from = {s: dijkstra(nadj, nadj, s) for s in nodes}
    k = len(odd)
    D = [[INF] * k for _ in range(k)]
    for i, s in enumerate(odd):
        for j, t in enumerate(odd):
            if t in dist_from[s]:
                D[i][j] = dist_from[s][t]
    costdp, _ = matching_dp(D, list(range(k)))
    return int(costdp[(1 << k) - 1])


def _minimum_tjoin_cardinality(
    nodes: Sequence[str], edges: Sequence[Edge], odd: Tuple[str, ...]
) -> int:
    """Smallest possible number of distinct edges in any parity-closing set.

    Unweighted all-pairs shortest paths + minimum perfect matching on the
    odd vertices.  Connected graphs guarantee a finite answer.
    """
    adj = adjacency(nodes, edges)
    k = len(odd)
    D = []
    for s in odd:
        dist = {s: 0}
        dq = deque([s])
        while dq:
            x = dq.popleft()
            for y, _ in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1
                    dq.append(y)
        D.append([dist.get(t, INF) for t in odd])
    costdp, _ = matching_dp(D, list(range(k)))
    return int(costdp[(1 << k) - 1])


def audit(
    nodes: Sequence[str],
    raw_edges: Sequence[dict],
    start: Optional[str],
    repeat_cap: Optional[int] = None,
) -> AuditResult:
    nodes, edges = validate_input(nodes, raw_edges, start)
    adj = adjacency(nodes, edges)

    comps = connected_components(nodes, adj)
    if len(comps) > 1:
        raise AuditError(
            "管网不连通，存在多个连通分量："
            + "；".join("{" + ",".join(c) + "}" for c in comps),
            ("edges", "nodes"),
            [{"field": "edges"}],
        )

    degree = {n: 0 for n in nodes}
    for e in edges:
        degree[e.u] += 1
        degree[e.v] += 1
    odd = tuple(sorted(n for n in nodes if degree[n] % 2 == 1))
    total_length = sum(e.length for e in edges)
    m = len(edges)

    # ------------------------------------------------------------------
    # Budget mode: exact minimization over parity-closing sets with at most
    # ``repeat_cap`` distinct duplicated edges.  This is *not* a filter on
    # the unconstrained optimum -- a longer but thinner solution may win.
    # ------------------------------------------------------------------
    if repeat_cap is not None:
        unconstrained_added = 0 if not odd else _unconstrained_added(
            nodes, edges, odd
        )
        if not odd:
            opt = ConstrainedOptimum(
                added=0, count=1, canonical_mask=0, in_all=0, in_any=0
            )
        else:
            opt = solve_cardinality_constrained(nodes, edges, odd, repeat_cap)
            if opt is None:
                min_card = _minimum_tjoin_cardinality(nodes, edges, odd)
                raise BudgetInfeasible(
                    repeat_cap, min_card, unconstrained_added
                )
        return _build_result(
            nodes=nodes, edges=edges, start=start, odd=odd, comps=comps,
            total_length=total_length,
            added_length=opt.added,
            optimal_count=opt.count,
            canonical_mask=opt.canonical_mask,
            in_all=opt.in_all, in_any=opt.in_any, m=m,
            repeat_cap=repeat_cap,
            repeat_count=opt.canonical_mask.bit_count(),
            unconstrained_added=unconstrained_added,
        )

    if not odd:
        empty: FrozenSet[int] = frozenset()
        multiplicity = tuple(1 for _ in edges)
        route = euler_circuit(nodes, edges, start, multiplicity)
        return AuditResult(
            nodes=nodes,
            edges=edges,
            start=start,
            odd_vertices=odd,
            components=tuple(tuple(c) for c in comps),
            total_length=total_length,
            added_length=0,
            optimal_count=1,
            canonical_set=empty,
            bit_vector="0" * m,
            classification={i: "never" for i in range(m)},
            multiplicity=multiplicity,
            route=tuple(route),
        )

    # shortest distances from every vertex
    nadj = build_nadj(nodes, edges)
    dist_from: Dict[str, dict] = {s: dijkstra(nodes, nadj, s) for s in nodes}

    k = len(odd)
    D = [[INF] * k for _ in range(k)]
    for i, s in enumerate(odd):
        for j, t in enumerate(odd):
            if t in dist_from[s]:
                D[i][j] = dist_from[s][t]

    costdp, _ = matching_dp(D, list(range(k)))
    full = (1 << k) - 1
    optimum = costdp[full]

    # distinct optimal duplicate sets, exact
    opt_masks = enumerate_optimal_tjoins(odd, dist_from, nadj, costdp, D)
    total_count = len(opt_masks)

    # canonical: 0 preferred at the earliest edge index => smallest binary
    # number with edge 0 as most significant bit
    canonical_mask = min(
        opt_masks, key=lambda mm: sum(1 << (m - 1 - i) for i in range(m) if mm >> i & 1)
    )

    in_all = (1 << m) - 1
    in_any = 0
    for mm in opt_masks:
        in_all &= mm
        in_any |= mm

    return _build_result(
        nodes=nodes, edges=edges, start=start, odd=odd, comps=comps,
        total_length=total_length,
        added_length=optimum,
        optimal_count=total_count,
        canonical_mask=canonical_mask,
        in_all=in_all, in_any=in_any, m=m,
    )
