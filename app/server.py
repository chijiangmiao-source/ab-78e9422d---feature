"""HTTP API for the pipe-network route audit."""

from __future__ import annotations

import os
from typing import Any, Dict

from .solver import (
    AuditError,
    BudgetInfeasible,
    Edge,
    RouteStep,
    audit,
    parse_repeat_cap,
)


def _serialize(nodes, edges: list[Edge], result) -> Dict[str, Any]:
    edge_objs = [
        {
            "id": e.eid,
            "u": e.u,
            "v": e.v,
            "length": e.length,
            "index": e.index,
            "classification": result.classification[e.index],
            "duplicated": e.index in result.canonical_set,
            "copies": result.multiplicity[e.index],
        }
        for e in edges
    ]
    route = [
        {
            "seq": i + 1,
            "edgeIndex": st.edge_index,
            "edgeId": st.edge_id,
            "from": st.frm,
            "to": st.to,
            "length": st.length,
            "copy": st.duplicate_no,
        }
        for i, st in enumerate(result.route)
    ]
    # positions at which each edge occurs in the route, for highlighting
    positions: Dict[int, list] = {i: [] for i in range(len(edges))}
    for i, st in enumerate(result.route):
        positions[st.edge_index].append(i + 1)

    payload = {
        "ok": True,
        "nodes": nodes,
        "edges": edge_objs,
        "start": result.start,
        "oddVertices": list(result.odd_vertices),
        "totalLength": result.total_length,
        "addedLength": result.added_length,
        "optimalCount": result.optimal_count,
        "canonicalVector": result.bit_vector,
        "canonicalEdges": [
            edges[i].eid for i in sorted(result.canonical_set)
        ],
        "route": route,
        "positions": {str(k): v for k, v in positions.items()},
        "eulerian": result.is_eulerian,
    }
    if result.budget_enabled:
        payload["repeatCap"] = result.repeat_cap
        payload["repeatCount"] = result.repeat_count
        payload["unconstrainedAddedLength"] = result.unconstrained_added_length
        payload["addedLengthDelta"] = result.added_length_delta
        payload["budgetFeasible"] = True
    return payload


def run_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pure logic entry: validate + audit, return a response dict."""
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    start = payload.get("start")
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    try:
        repeat_cap = (
            parse_repeat_cap(payload["repeatCap"])
            if "repeatCap" in payload
            else None
        )
        result = audit(nodes, edges, start, repeat_cap=repeat_cap)
    except BudgetInfeasible as exc:
        return {
            "ok": True,
            "budgetFeasible": False,
            "repeatCap": exc.cap,
            "minimumRepeatCount": exc.min_required,
            "unconstrainedAddedLength": exc.unconstrained_added,
        }
    except AuditError as exc:
        return {
            "ok": False,
            "error": exc.message,
            "fields": list(exc.fields),
            "locations": exc.locations,
        }
    return _serialize(result.nodes, result.edges, result)


def create_app():
    from flask import Flask, jsonify, request, send_from_directory

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app = Flask(__name__, static_folder=static_dir, static_url_path="")

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/audit")
    def api_audit():
        payload = request.get_json(silent=True) or {}
        return jsonify(run_audit(payload))

    return app


app = create_app()
