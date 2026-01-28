import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os, sys, json, argparse, pickle, re
from pathlib import Path
import logging
import numpy as np
import openpyxl
import unicodedata


logging.basicConfig(level=logging.INFO)

COMP_SUFFIX = r"(?:c|m|e|x|l)"  # add more if your data has them

_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "κ": "kappa", "λ": "lambda", "μ": "mu",
    "ω": "omega",
}

def normalize_key(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s)
    for k, v in _GREEK.items():
        s = s.replace(k, v)
    s = s.lower()

    # keep letters/numbers, turn everything else into spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    return [str(x)]


def as_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"true","t","1","y","yes"}

def pick_irreversible(row):
    # prefer curated/filtered if present
    for col in ["Fil.Irreversible", "Irreversible", "Irreversible_Human1", "Irreversible.Old"]:
        if col in row and str(row[col]).strip() != "":
            return as_bool(row[col])
    return False



def parse_met_list(field: str):
    """
    Parses strings like:
      'C00007_c + C00001_c'
      '2 C00027_c <=> C00007_c + 2 C00001_c'
    into:
      ['C00027_c', 'C00007_c', 'C00001_c']
    """
    if field is None or (pd.isna(field)):
        return []

    s = str(field)
    # Pull ONLY valid CIDs with compartment suffix
    mets = re.findall(rf"(C\d{{5}}_{COMP_SUFFIX})", s)
    return [m.strip() for m in mets if m and m.strip()]
    


def parse_gene_list(field: str):
    """
    Parses gene symbols field like:
      'CAT;ABTB2' or 'PDHA1;PDHB;DLAT;DLD'
    """
    if field is None or (isinstance(field, float) and pd.isna(field)):
        return []
    raw = re.split(r"[;,|/]", str(field))
    return [g.strip() for g in raw if g and g.strip()]


def build_mapping_index(met_df: pd.DataFrame, alt_names_net: dict) -> dict:
    """
    Build normalized_name -> base_cid (C00024)
    Uses canonical Name and synonym names
    """
    if "Cid" not in met_df.columns or "Name" not in met_df.columns:
        raise ValueError("Metabolite table must contain columns: Cid and Name")

    syn = {}

    def add_key(k: str, cid: str):
        k = normalize_key(k)
        if not k:
            return
        # This is important: prevents garbage like "o", "a", "t"
        if len(k) < 2:
            return
        syn.setdefault(k, cid)


    # canonical or primary names
    for cid, name in zip(met_df["Cid"], met_df["Name"]):
        if cid is None or (isinstance(cid, float) and pd.isna(cid)):
            continue
        base_cid = str(cid).strip()

        if not base_cid.startswith("C"):
            # skip weird ids
            continue

        if name is None or (isinstance(name, float) and pd.isna(name)):
            # skip weird names
            continue

        add_key(str(name), base_cid)


    # Alternate names
    for k, v in alt_names_net.items():
        current_cid = str(k).strip()
        if not current_cid.startswith("C"):
            continue

        for main_name in (v.get("NAME") or []):
            add_key(main_name, current_cid)

        for alt_name in (v.get("SYNONYMS") or []):
            add_key(alt_name, current_cid)

    return syn



def extract_all_genes_from_reactions(rx_df: pd.DataFrame) -> set:
    """
    Collect all unique gene symbols that appear in the reaction table.
    Uses Final.Symbol (or whatever column you use for gene mapping).
    """
    genes = set()

    col_genes = "Final.Symbol"
    if col_genes not in rx_df.columns:
        raise ValueError(f"Reaction table missing required gene column: {col_genes}")

    for v in rx_df[col_genes].tolist():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        # same splitting logic you used earlier
        raw = re.split(r"[;,|/]", str(v))
        for g in raw:
            g = g.strip()
            if g:
                genes.add(g.upper())
    return genes


def compute_met_importance(met_df: pd.DataFrame) -> dict:
    """
    Returns base_cid -> importance dict with S_final and supporting fields.
    Expected columns (based on your snapshot):
      - Cid (base CID)
      - F3
      - KEGG_in_degree (or KEGG_in_degree.1)
      - KEGG_out_degree (or KEGG_out_degree.1)
    """
    cid_col = "Cid"
    if cid_col not in met_df.columns:
        raise ValueError(f"Missing {cid_col} in metabolite table")

    # try both naming conventions
    in_col  = "KEGG_in_degree"  if "KEGG_in_degree"  in met_df.columns else "KEGG_in_degree.1"
    out_col = "KEGG_out_degree" if "KEGG_out_degree" in met_df.columns else "KEGG_out_degree.1"

    if in_col not in met_df.columns or out_col not in met_df.columns:
        raise ValueError("Missing KEGG_in/out_degree columns in metabolite table")

    def score_deg(x: float) -> int:
        x = float(x or 0)
        return (2 if x > 20 else 0) + (2 if x > 10 else 0) + (1 if x > 5 else 0) + (1 if x > 0 else 0)

    imp = {}
    for _, r in met_df.iterrows():
        cid = str(r[cid_col]).strip() if pd.notna(r[cid_col]) else ""
        if not cid:
            continue

        F3 = float(r.get("F3") or 0.0)
        kin = float(r.get(in_col) or 0.0)
        kout = float(r.get(out_col) or 0.0)

        S_in = score_deg(kin)
        S_out = score_deg(kout)
        S_final = (S_in + S_out) * abs(F3 - 1.0)

        imp[cid] = {
            "S_in": S_in,
            "S_out": S_out,
            "S_final": float(S_final),
            "F3": float(F3),
            "kegg_in_degree": kin,
            "kegg_out_degree": kout,
            "name": (str(r.get("Name") or "").strip() or None),
        }

    return imp



def build_metabolic_graph(rx_df: pd.DataFrame, met_df: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Graph schema:
      Nodes:
        - reaction_ids: BDRL_id
        - metabolite_id: C00024_m
        - gene: PDHA1
      Edges (directed):
        - metabolite -> reaction (type=consumes)
        - reaction -> metabolite (type=produces)
        - gene -> reaction (type=catalyzes)
    """
    # base CID -> display name
    cid_to_name = {}
    if "Cid" in met_df.columns and "Name" in met_df.columns:
        for cid, name in zip(met_df["Cid"], met_df["Name"]):
            if cid is None or (isinstance(cid, float) and pd.isna(cid)):
                continue
            base = str(cid).strip()
            cid_to_name[base] = None if (name is None or (isinstance(name, float) and pd.isna(name))) else str(name).strip()
            # print("#$#$#$$##$$$#$$# Inside cid_to_name block", cid_to_name[base], base)
    G = nx.MultiDiGraph()

    # Column fallbacks
    col_rid = "BDRL_id" # to do maybe remove BDRL prefix so it does not leak during LLM generation
    col_genes = "Final.Symbol"
    col_in = "C_in" if "C_in" in rx_df.columns else "C_in_v1.1" 
    col_out = "C_out" if "C_out" in rx_df.columns else "C_out_v1.1"
    col_irrev = "Irreversible" if "Irreversible" in rx_df.columns else "Fil.Irreversible"
    col_comp = "Final_SL_abbr" if "Final_SL_abbr" in rx_df.columns else "Final_SL"
    print(col_in, col_out, col_irrev, col_comp)

    imp = compute_met_importance(met_df)
    
    for _, row in rx_df.iterrows():
        rid = str(row.get(col_rid, "")).strip()

        # Some keggid have multiple ids separated by _

        if not rid:
            continue

        # irreversible = _as_bool(row.get(col_irrev, False)) # returns false if missing or NA
        irreversible = pick_irreversible(row)
        compartment = row.get(col_comp, None)

        # Reaction node
        G.add_node(
            rid,
            kind="reaction",
            irreversible=irreversible,
            compartment=None if (isinstance(compartment, float) and pd.isna(compartment)) else compartment,
            pathway=row.get("Pathway", None), # to do seperated by ;
            kegg_id=row.get("KEGG_ID", None), # to do some kegg_ids have multiple ids separated by _ what are these mean
        )

        # Gene nodes + edges
        genes = parse_gene_list(row.get(col_genes, ""))
        for g in genes:
            g = g.strip().upper()
            if not g:
                continue
            G.add_node(g, kind="gene")
            G.add_edge(g, rid, type="catalyzes")

        # Metabolite nodes + edges
        ins = parse_met_list(row.get(col_in, ""))
        outs = parse_met_list(row.get(col_out, ""))
        print('\n\n Next reaction', rid)

        for m_id in ins:
            m_id = str(m_id).strip()
            base_cid = m_id.split("_")[0]
            base = base_cid  # e.g. "C00024"
            meta_imp = imp.get(base, {})
            print('Inside building_metabolic_network -> m_id in ins', m_id, base, cid_to_name.get(base), meta_imp, meta_imp.get("name"))
            G.add_node(
                m_id,
                kind="metabolite",
                base_cid=base,
                name=cid_to_name.get(base) or meta_imp.get("name"),
                compartment=m_id.split("_")[-1],
                S_final=meta_imp.get("S_final", 0.0),
                F3=meta_imp.get("F3", None),
                kegg_in_degree=meta_imp.get("kegg_in_degree", None),
                kegg_out_degree=meta_imp.get("kegg_out_degree", None),
            )
            G.add_edge(m_id, rid, type="consumes")

        for m_id in outs:
            m_id = str(m_id).strip()
            base_cid = m_id.split("_")[0]
            base = base_cid  # e.g. "C00024"
            meta_imp = imp.get(base, {})
            print('Inside building_metabolic_network -> m_id in outs', m_id, base, cid_to_name.get(base), meta_imp, meta_imp.get("name"))

            G.add_node(
                m_id,
                kind="metabolite",
                base_cid=base,
                name=cid_to_name.get(base) or meta_imp.get("name"),
                compartment=m_id.split("_")[-1],
                S_final=meta_imp.get("S_final", 0.0),
                F3=meta_imp.get("F3", None),
                kegg_in_degree=meta_imp.get("kegg_in_degree", None),
                kegg_out_degree=meta_imp.get("kegg_out_degree", None),
            )
            G.add_edge(rid, m_id, type="produces")

    return G


def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--reactions", required=True)
    # ap.add_argument("--metabolites", required=True)
    # ap.add_argument("--outdir", required=True)
    # ap.add_argument("--met_sheet", default=0)
    # args = ap.parse_args()

    outdir_name = "/home/ubuntu/WebDemoKHGRAG/metabolite_network_integration/metabolic_graph_outputs"
    metabolites_file="/home/ubuntu/WebDemoKHGRAG/metabolite_network_integration/total_human_C_stat_check_working7.xlsx"
    reactions_file = "/home/ubuntu/WebDemoKHGRAG/metabolite_network_integration/20260116_core_reactions_012026_M_v1.2.csv"
    met_sheet = 0
    alternate_names_file = "/home/ubuntu/WebDemoKHGRAG/metabolite_network_integration/BDRL_metab_synonyms_2026.json"

    outdir = Path(outdir_name).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rx_df = pd.read_csv(reactions_file)
    met_df = pd.read_excel(metabolites_file, sheet_name=met_sheet)
    with open(alternate_names_file, "r", encoding="utf-8") as f:
        alt_names_net = json.load(f)

    name_id_mapping = build_mapping_index(met_df, alt_names_net)
    G = build_metabolic_graph(rx_df, met_df)

    (outdir / "name_id_met_mapping.json").write_text(json.dumps(name_id_mapping, indent=2), encoding="utf-8")
    with open(outdir / "met_graph.pkl", "wb") as f:
        pickle.dump(G, f)

    all_genes = extract_all_genes_from_reactions(rx_df)
    (outdir / "gene_symbols.json").write_text(
        json.dumps(sorted(all_genes), indent=2),
        encoding="utf-8",
    )

    print("Saved:")
    print(" -", outdir / "met_graph.pkl")
    print(" -", outdir / "name_id_met_mapping.json")
    print(" -", outdir / "gene_symbols.json", f"({len(all_genes)} genes)")
    print("Graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")


if __name__ == "__main__":
    main()


