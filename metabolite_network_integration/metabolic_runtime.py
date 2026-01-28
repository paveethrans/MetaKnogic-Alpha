import json
import pickle
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from rapidfuzz import process, fuzz
import networkx as nx




SHORT_OK = {
    "o2", "h2o", "co2", "nh3", "h+", "atp", "adp", "amp",
    "nad", "nadh", "nadp", "nadph", "coa",
}

GREEK_MAP = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "κ": "kappa",
}

def normalize_key(text: str) -> str:
    """
    Single normalization used everywhere for metabolite name matching.
    - lowercase
    - translate a few greek letters
    - keep letters/digits/+/- and spaces
    - collapse whitespace
    """
    t = (text or "").strip().lower()
    for g, repl in GREEK_MAP.items():
        t = t.replace(g, repl)
    # keep a-z 0-9 space + -
    t = re.sub(r"[^a-z0-9\+\-\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse_entities_block(text: str) -> dict:
    """
    Extract entity lines like:
      GENE: IDH1
      MET: 2-hydroxyglutarate
      MET_SYNONYM: 2-HG
    Returns dict with lists.
    """
    out = {"genes": [], "mets": [], "paths": [], "procs": [], "raw": []}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        out["raw"].append(line)

        m = re.match(r"^(GENE|MET|MET_SYNONYM|PATHWAY|PROCESS)\s*:\s*(.+)$", line, flags=re.I)
        if not m:
            continue
        tag = m.group(1).upper()
        val = m.group(2).strip()

        if tag == "GENE":
            out["genes"].append(val)
        elif tag in {"MET", "MET_SYNONYM"}:
            out["mets"].append(val)
        elif tag == "PATHWAY":
            out["paths"].append(val)
        elif tag == "PROCESS":
            out["procs"].append(val)
    return out



# def extract_gene_hits(query: str) -> List[str]:
#     # MVP heuristic; later replace with HGNC alias map
#     toks = re.findall(r"\b[A-Z0-9]{2,10}\b", query or "")
#     stop = {"AND", "OR", "THE", "WITH", "IN", "ON", "TO"}
#     return [t for t in toks if t not in stop]


def extract_cid_hits(query: str) -> List[str]:
    # allow users to type C00024 or C00024_m
    base = re.findall(r"\b(C\d{5})\b", query or "")
    comp = re.findall(r"\b(C\d{5}_(?:c|m|e|x|l))\b", query or "")
    return list(dict.fromkeys(comp + base))



TOKEN_RE = re.compile(r"[A-Za-z0-9\-]+")

def tokenize_query(q: str) -> list[str]:
    """
    Splits text into tokens like:
      'IDO1', 'HLA-DRA', 'tryptophan', 'PDCD1'
    Keeps hyphens because many gene symbols have them.
    """
    return TOKEN_RE.findall(q or "")



def expand_two_hops(G: nx.MultiDiGraph, seeds: List[str], max_nodes: int = 300) -> nx.MultiDiGraph:
    keep = set()
    frontier = set(s for s in seeds if s in G)

    keep |= frontier

    hop1 = set()
    for n in frontier:
        hop1 |= set(G.successors(n))
        hop1 |= set(G.predecessors(n))
    keep |= hop1

    hop2 = set()
    for n in hop1:
        hop2 |= set(G.successors(n))
        hop2 |= set(G.predecessors(n))
    keep |= hop2

    if len(keep) > max_nodes:
        keep = set(list(keep)[:max_nodes])

    return G.subgraph(keep).copy()


def ppr_subgraph_debug(
    G: nx.MultiDiGraph,
    seeds: List[str],
    alpha: float = 0.85,
    topk: int = 250,
    hub_degree_cap: int = 200,
) -> nx.MultiDiGraph:
    # Project MultiDiGraph -> weighted DiGraph
    H = nx.DiGraph()
    deg = dict(nx.Graph(G).degree())

    for u, v, data in G.edges(data=True):
        et = data.get("type", "")
        w = 1.0
        if et == "catalyzes":
            w = 0.3

        # penalize hubs
        if deg.get(u, 0) > hub_degree_cap or deg.get(v, 0) > hub_degree_cap:
            w *= 0.1

        if H.has_edge(u, v):
            H[u][v]["weight"] = max(H[u][v]["weight"], w)
        else:
            H.add_edge(u, v, weight=w)

    p = {n: 0.0 for n in H.nodes()}
    for s in seeds:
        if s in p:
            p[s] = 1.0
    z = sum(p.values())
    if z == 0:
        return G.subgraph([]).copy()
    for k in p:
        p[k] /= z

    scores = nx.pagerank(H, alpha=alpha, personalization=p, weight="weight")
    top_nodes = sorted(scores, key=scores.get, reverse=True)[:topk]
    return G.subgraph(top_nodes).copy()



def keep_node(G, n, s_final_min=1.0):
    a = G.nodes[n]
    if a.get("kind") != "metabolite":
        return True
    s = float(a.get("S_final") or 0.0)
    return s >= s_final_min


def prune_metabolites(subG: nx.MultiDiGraph, s_final_min: float = 1.0, drop_currency: bool = True) -> nx.MultiDiGraph:
    keep = set()
    for n, a in subG.nodes(data=True):
        if a.get("kind") != "metabolite":
            keep.add(n)
            continue

        base = a.get("base_cid")

        s = float(a.get("S_final") or 0.0)
        if s >= s_final_min:
            keep.add(n)

    # also keep reactions that remain connected to kept metabolites/genes
    # by re-adding any reaction that touches a kept node
    for u, v in subG.edges():
        if u in keep or v in keep:
            keep.add(u); keep.add(v)

    return subG.subgraph(keep).copy()





class MetabolicNetwork:
    def __init__(self, graph_path: str, name_id_metabolite_mapping: str, gene_symbols_path: str,mode: str = "two_hop"):
        self.graph_path = str(Path(graph_path).resolve())
        self.name_id_metabolite_mapping = str(Path(name_id_metabolite_mapping).resolve())
        self.gene_symbols_path = str(Path(gene_symbols_path).resolve())
        self.mode = mode

        with open(self.graph_path, "rb") as f:
            self.G: nx.MultiDiGraph = pickle.load(f)

        self.syn: Dict[str, str] = json.loads(Path(self.name_id_metabolite_mapping).read_text(encoding="utf-8"))

        # Load curated gene symbols (uppercase)
        genes = json.loads(Path(self.gene_symbols_path).read_text(encoding="utf-8"))
        self.gene_set = {str(g).upper() for g in genes if str(g).strip()}


        # quick index: base CID -> all compartment metabolite node ids present
        self.base_to_nodes: Dict[str, List[str]] = {}
        for n, attr in self.G.nodes(data=True):
            if attr.get("kind") == "metabolite":
                base = attr.get("base_cid")
                if base:
                    self.base_to_nodes.setdefault(base, []).append(n)


        # Build filtered synonym key index for exact matching (avoid 1-letter junk)
        self.syn_key_to_cid: Dict[str, str] = {}
        for k, cid in self.syn.items():
            kk = normalize_key(k)
            if not kk:
                continue
            if len(kk) <= 2 and kk not in SHORT_OK:
                continue
            # keep first mapping; don't overwrite
            if kk not in self.syn_key_to_cid:
                self.syn_key_to_cid[kk] = cid

        self._syn_keys = list(self.syn_key_to_cid.keys())

    # Convert names to based nodes and then in out [] add compartment nodes and return
    def _metabolite_hits(self, query: str) -> List[str]:
        # Prefer structured entity block if present
        ent = parse_entities_block(query)
        candidates = ent["mets"][:]  # metabolite names + synonyms from rewrite

        # If no structured lines exist, fallback to using the whole string as ONE candidate
        # (still exact/fuzzy; no substring scanning)
        if not candidates:
            candidates = [query]

        hits_base = set()

        # 1) explicit CIDs in text always win
        for cid in extract_cid_hits(query):
            if cid.startswith("C") and len(cid) == 6:
                hits_base.add(cid)
            elif cid.startswith("C") and "_" in cid:
                # already compartment-specific node
                return [cid.strip()]

        # 2) exact match on normalized key
        for c in candidates:
            ck = normalize_key(c)
            if not ck:
                continue
            cid = self.syn_key_to_cid.get(ck)
            if cid:
                hits_base.add(cid)

        # 3) near-exact fuzzy match ONLY if still empty
        if not hits_base:
            for c in candidates:
                ck = normalize_key(c)
                if not ck:
                    continue
                # avoid fuzzy on very short strings (prevents garbage matches)
                if len(ck) <= 3 and ck not in SHORT_OK:
                    continue
                best = process.extractOne(
                    ck,
                    self._syn_keys,
                    scorer=fuzz.WRatio,
                )
                if best:
                    key_match, score, _ = best
                    if score >= 92:  # strict threshold
                        hits_base.add(self.syn_key_to_cid[key_match])

        # expand base CID -> all compartment nodes
        out = []
        for base in sorted(hits_base):
            out.extend([x.strip() for x in self.base_to_nodes.get(base, [])])
        return out


    def _gene_hits(self, query: str) -> List[str]:
        hits = []
        seen = set()

        for tok in tokenize_query(query):
            g = tok.upper()

            # exact match to curated gene list
            if g in self.gene_set and g in self.G:
                if g not in seen:
                    seen.add(g)
                    hits.append(g)

        return hits


    def _seed_nodes(self, query: str) -> List[str]:
        seeds = []

        # gene hits
        seeds.extend(self._gene_hits(query))

        # metabolite hits
        seeds.extend([m for m in self._metabolite_hits(query) if m in self.G]) # With compartmental values

        # de-dup preserving order
        seen = set()
        uniq = []
        for s in seeds:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq

    def _format_context(self, subG: nx.MultiDiGraph, seeds: List[str], top_reactions: int = 12) -> Tuple[str, Dict[str, Any]]:
        reactions = []
        for n, attr in subG.nodes(data=True):
            if attr.get("kind") == "reaction":
                reactions.append((n, attr))

        # rank reactions crudely: how many seed-adjacent edges touch it
        seed_set = set(seeds)

        def met_obj(G, node_id):
            a = G.nodes[node_id]
            return {
                "node": node_id,
                "base_cid": a.get("base_cid"),
                "name": a.get("name"),
                "comp": a.get("compartment"),
                "S_final": float(a.get("S_final") or 0.0),
            }

        def met_str(x):
            return f"{x['name']} ({x['node']})"

        def met_score(subG, met_node):
            return float(subG.nodes[met_node].get("S_final") or 0.0)

        def reaction_support_score(ins, outs):
            return sum(met_score(subG, m) for m in ins) + sum(met_score(subG, m) for m in outs)

        def met_label(G, m):
            a = G.nodes[m]
            nm = a.get("name") or a.get("base_cid") or m
            return f"{nm} ({m})"

        def r_score(rid: str) -> int:
            s = 0
            for u, v, data in subG.in_edges(rid, data=True):
                if u in seed_set:
                    s += 2
                if data.get("type") in {"consumes", "produces"}:
                    s += 1
            for u, v, data in subG.out_edges(rid, data=True):
                if v in seed_set:
                    s += 2
                if data.get("type") in {"consumes", "produces"}:
                    s += 1
            return s



        reactions_sorted = sorted(reactions, key=lambda x: (r_score(x[0]), reaction_support_score(
            sorted({u for u, v, d in subG.in_edges(x[0], data=True) if d.get("type") == "consumes"}),
            sorted({v for u, v, d in subG.out_edges(x[0], data=True) if d.get("type") == "produces"}))), reverse=True)[:top_reactions]


        irr = sum(1 for rid, a in reactions_sorted if a.get("irreversible"))
        comps = sorted({str(a.get("compartment") or "") for _, a in reactions_sorted if a.get("compartment")})

        lines = []
        lines.append("Metabolic network context (curated; use as mechanistic grounding):")
        lines.append(f"- Seeds used: {seeds if seeds else 'none'}")
        lines.append(f"- Subgraph size: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")
        lines.append(f"- Reactions shown: {len(reactions_sorted)} (irreversible among shown: {irr})")
        if comps:
            lines.append(f"- Compartments in shown set: {', '.join([c for c in comps if c])}")
        lines.append("")
        lines.append("Top reactions (respect directionality and compartments; do not invent transport steps):")

        out_json = {"seeds": seeds, "reactions": []}

        for rid, a in reactions_sorted:
            # genes = incoming catalyzes
            genes = sorted({u for u, v, d in subG.in_edges(rid, data=True) if d.get("type") == "catalyzes"})
            # inputs = incoming consumes from metabolites
            ins = sorted({u for u, v, d in subG.in_edges(rid, data=True) if d.get("type") == "consumes"})
            # outputs = outgoing produces to metabolites
            outs = sorted({v for u, v, d in subG.out_edges(rid, data=True) if d.get("type") == "produces"})

            inputs = [met_obj(subG, m) for m in ins]
            outputs = [met_obj(subG, m) for m in outs]

            support = reaction_support_score(ins, outs)
            score = r_score(rid)

            irrev = "IRREV" if a.get("irreversible") else "REV/UNK"
            comp = a.get("compartment") or "NA"

            pathway = a.get("Pathway") or a.get("pathway")  # supports either key
            module_ids = a.get("KEGG_module_ids") or a.get("kegg_module_ids")
            module_names = a.get("KEGG_module_names") or a.get("kegg_module_names")

            lines.append(f"* {rid} [{comp}] [{irrev}] score={score} support={support:.2f}")
            if genes:
                lines.append(f"  genes: {', '.join(genes[:15])}")
            if inputs:
                lines.append(f"  in: {', '.join(met_str(x) for x in inputs[:20])}")
            if outputs:
                lines.append(f"  out: {', '.join(met_str(x) for x in outputs[:20])}")
            if pathway:
                lines.append(f"  pathway: {pathway}")


            out_json["reactions"].append({
                "rid": rid,
                "score": int(score),
                "support": float(support),
                "irreversible": bool(a.get("irreversible")),
                "compartment": comp,
                "pathway": pathway,
                "module_ids": module_ids,
                "module_names": module_names,
                "genes": genes,
                "in": ins,
                "out": outs,
                "in_mets": inputs,
                "out_mets": outputs,
            })

        pathway_to_rids = defaultdict(list)

        for rid, a in reactions_sorted:
            pw = a.get("Pathway") or a.get("pathway") or "Unknown"
            for p in [x.strip() for x in str(pw).split(";") if x.strip()]:
                pathway_to_rids[p].append(rid)

        top_pathways = sorted(pathway_to_rids, key=lambda p: len(pathway_to_rids[p]), reverse=True)[:4]

        out_json["pathway_views"] = [
            {
                "label": p,
                "why_selected": f"{len(pathway_to_rids[p])} retrieved reactions in this pathway",
                "reactions": pathway_to_rids[p][:30],
            }
            for p in top_pathways
        ]

        if top_pathways:
            lines.append("")
            lines.append("Pathway perspectives (for multi-view reasoning):")
            for p in top_pathways:
                lines.append(f"- {p} ({len(pathway_to_rids[p])} reactions)")
        
        return "\n".join(lines), out_json

    def get_context(self, query: str, top_reactions: int = 20) -> Tuple[str, Dict[str, Any]]:
        seeds = self._seed_nodes(query)
        if not seeds:
            ctx = "Metabolic network context: no confident gene/metabolite hits found in the curated network for this query."
            return ctx, {"seeds": [], "reactions": []}

        if self.mode == "ppr":
            subG = ppr_subgraph_debug(self.G, seeds)
        else:
            subG = expand_two_hops(self.G, seeds)

        subG = prune_metabolites(subG, s_final_min=1.0, drop_currency=True)

        return self._format_context(subG, seeds, top_reactions=top_reactions)
