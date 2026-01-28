import math
import inspect
from typing import Dict, Any, List


def build_local_subgraph_json(
    node_meta: List[Dict[str, Any]],
    edge_meta: List[Dict[str, Any]],
    max_entities: int = 10,
    max_hyperedges: int = 10,
    chunks_data: dict = None,
) -> Dict[str, Any]:
    """
    node_meta: list of dicts like:
      { "id": ..., "role": "entity" | "hyperedge", "weight": float, "entity_type": ..., "description": ... }

    edge_meta: list of dicts like:
      { "src": ..., "tgt": ..., "weight": float }

    Returns a Cytoscape-style JSON:
      {
        "nodes": [...],
        "edges": [...]
      }
    """
    # --- split entities vs hyperedges ---
    entities = [n for n in node_meta if n.get("role") == "entity"]
    hyperedges = [n for n in node_meta if n.get("role") == "hyperedge"]

    # fallback weights if missing
    for n in entities + hyperedges:
        n.setdefault("weight", 1.0)

    # sort by weight desc and keep top-N
    entities_sorted = sorted(entities, key=lambda n: n["weight"], reverse=True)[:max_entities]
    hyperedges_sorted = sorted(hyperedges, key=lambda n: n["weight"], reverse=True)[:max_hyperedges]

    # keep only ids in this reduced subgraph
    keep_ids = {n["id"] for n in entities_sorted + hyperedges_sorted}

    # filter edges where both endpoints are in keep_ids
    filtered_edges = [
        e for e in edge_meta
        if e.get("src") in keep_ids and e.get("tgt") in keep_ids
    ]

    # --- normalize weights for styling (node size / edge width) ---
    def _normalize(values, min_size=20, max_size=60):
        if not values:
            return {}
        vmin, vmax = min(values), max(values)
        if math.isclose(vmin, vmax):
            # all same → give mid-size
            return {v: (min_size + max_size) / 2 for v in values}
        norm = {}
        for v in values:
            t = (v - vmin) / (vmax - vmin)  # 0–1
            norm[v] = min_size + t * (max_size - min_size)
        return norm

    node_weights = [n["weight"] for n in entities_sorted + hyperedges_sorted]
    node_size_map = _normalize(node_weights, min_size=20, max_size=100)

    edge_weights = [e.get("weight", 1.0) for e in filtered_edges]
    edge_width_map = _normalize(edge_weights, min_size=1, max_size=10)

    # --- build Cytoscape-style elements ---
    cy_nodes = []
    for n in entities_sorted + hyperedges_sorted:
        w = n["weight"]
        label = n["id"]
        paper_id = n.get("paper_id", "Unknown") # pmc paepr id
        # Clean up label a bit for display
        if isinstance(label, str) and label.startswith('"') and label.endswith('"'):
            label = label.strip('"')
        if isinstance(label, str) and label.startswith("<hyperedge>"):
            # collapse long text; front-end can show full text in tooltip
            label = "Evidence: " + label[len("<hyperedge>"):][:80] + "..."

        cy_nodes.append({
            "data": {
                "id": n["id"],
                "label": label,
                "role": n.get("role", "entity"),
                "weight": w,
                "size": node_size_map.get(w, 35),
                "entity_type": n.get("entity_type"),
                "description": n.get("description"),
                "paper_id": paper_id,
            }
        })

    cy_edges = []
    for idx, e in enumerate(filtered_edges):
        w = e.get("weight", 1.0)
        cy_edges.append({
            "data": {
                "id": f"e{idx}",
                "source": e["src"],
                "target": e["tgt"],
                "weight": w,
                "width": edge_width_map.get(w, 2),
            }
        })

    return {
        "nodes": cy_nodes,
        "edges": cy_edges,
    }


async def build_subgraph_for_query(
    rag,
    query_text: str,
    top_k: int = 5,
    hops: int = 1,
    max_entities: int = 10,
    max_hyperedges: int = 10,
    chunks_data: dict = None,
) -> Dict[str, Any]:
    """
    Uses entities_vdb to find seed entities, expands 1-hop in chunk_entity_relation_graph,
    fetches node / edge metadata, and returns Cytoscape JSON for the frontend.
    """

    # --- 1) Get seed entities from entities_vdb ---
    vdb = getattr(rag, "entities_vdb", None)
    if vdb is None:
        return {"nodes": [], "edges": []}

    qfunc = getattr(vdb, "query", None) or getattr(vdb, "search", None)
    if qfunc is None:
        return {"nodes": [], "edges": []}

    res = qfunc(query_text, top_k=top_k)
    if inspect.isawaitable(res):
        res = await res

    seeds = []
    for r in res:
        name = r.get("entity_name") or r.get("name") or r.get("id")
        if name:
            seeds.append(name)

    if not seeds:
        return {"nodes": [], "edges": []}

    # --- 2) Expand in graph (1-hop or more) ---
    if not hasattr(rag, "chunk_entity_relation_graph"):
        return {"nodes": [], "edges": []}

    graph = getattr(rag, "chunk_entity_relation_graph")
    nodes = set(seeds)
    edges = set()
    frontier = list(seeds)

    get_edges = getattr(graph, "get_node_edges", None)
    get_node = getattr(graph, "get_node", None)
    get_edge = getattr(graph, "get_edge", None)

    if get_edges is None or get_node is None or get_edge is None:
        return {"nodes": [], "edges": []}

    for _ in range(max(1, hops)):
        new_frontier = []
        for n in frontier:
            res_edges = get_edges(n)
            if inspect.isawaitable(res_edges):
                res_edges = await res_edges
            if not res_edges:
                continue

            for s, t in res_edges:
                edges.add((s, t))
                nodes.add(s)
                nodes.add(t)
                new_frontier.append(t)
        frontier = new_frontier

    # --- 3) Fetch metadata for nodes ---
    node_meta = []
    for n in nodes:
        md = get_node(n)
        if inspect.isawaitable(md):
            md = await md
        if md is None:
            md = {}
        role = md.get("role", "entity")
        weight = md.get("weight", 1.0)
        node_meta.append({
            "id": n,
            "role": role,
            "weight": weight,
            **{k: v for k, v in md.items() if k not in ("id", "role", "weight")},
        })

    # --- 4) Fetch metadata for edges ---
    edge_meta = []
    for s, t in edges:
        md = get_edge(s, t)
        if inspect.isawaitable(md):
            md = await md
        if md is None:
            md = {}
        weight = md.get("weight", 1.0)
        edge_meta.append({
            "src": s,
            "tgt": t,
            "weight": weight,
            **{k: v for k, v in md.items() if k not in ("src", "tgt", "weight")},
        })

    # --- 5) Convert to Cytoscape JSON with top-N filtering and weight-based sizing ---
    graph_json = build_local_subgraph_json(
        node_meta=node_meta,
        edge_meta=edge_meta,
        max_entities=max_entities,
        max_hyperedges=max_hyperedges,
        chunks_data=chunks_data,
    )

    return graph_json
