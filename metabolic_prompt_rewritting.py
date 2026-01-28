"""
Utilities for rewriting a *base RAG answer* into pathway-perspective reasoning,
using curated metabolic network context as:
1) mechanistic grounding,
2) constraint validator (directionality + compartments),
3) evidence linker (cite BDRL_id only when network-backed).

The main entrypoint is `rewrite_answer_with_metabolic_grounding`, which uses an LLM to:
- preserve the spirit/content of the base RAG synthesis,
- restructure into 2–4 pathway-level perspectives,
- add network-backed mechanistic steps where supported,
- explicitly flag what is NOT supported by the curated network,
- emit a structured JSON result + a final formatted answer string.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncOpenAI  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.1"


# -------------------------
# Dataclasses
# -------------------------

@dataclass
class MetabolicPerspective:
    label: str
    mechanistic_story: str
    network_backed_steps: List[str]
    evidence_genes: List[str]
    evidence_metabolites: List[str]  # prefer "Name (CID_comp)" strings
    evidence_reactions: List[str]    # BDRL_ids
    adds_beyond_literature: str
    limits_uncertainty: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintFinding:
    kind: str  # e.g., "irreversibility", "compartment", "missing_transport", "unsupported_claim"
    detail: str
    related_reactions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetabolicRewriteResult:
    final_answer: str
    perspectives: List[MetabolicPerspective]
    constraint_check: List[ConstraintFinding]
    metabolic_evidence_used: List[str]  # BDRL_ids only
    warnings: List[str]
    raw_json: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------
# Helpers
# -------------------------

def _read_api_key_from_file_or_env(
    *,
    api_key: Optional[str] = None,
    key_filename: str = "openai_key",
) -> Optional[str]:
    """
    Priority:
    1) explicit api_key argument
    2) OPENAI_API_KEY env var
    3) openai_key file next to this module
    """
    if api_key and api_key.strip():
        return api_key.strip()

    env = os.environ.get("OPENAI_API_KEY", "").strip()
    if env:
        return env

    key_path = Path(__file__).with_name(key_filename)
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()

    return None


def _truncate_block(s: str, max_chars: int = 12000, max_lines: int = 180) -> str:
    if not s:
        return ""
    lines = s.splitlines()
    s2 = "\n".join(lines[:max_lines])
    return s2[:max_chars]


def _coerce_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    v = str(value).strip()
    return [v] if v else []


def _convert_response_to_json(response: str) -> Dict[str, Any]:
    if not isinstance(response, str):
        raise ValueError("LLM response is not a string.")
    txt = response.strip()

    # direct JSON
    try:
        out = json.loads(txt)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass

    # pull first JSON object
    m = re.search(r"{.*}", txt, re.DOTALL)
    if m:
        try:
            out = json.loads(m.group(0))
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            pass

    raise ValueError("Unable to parse JSON from LLM response.")


def _reduce_met_json(
    met_json: Dict[str, Any],
    *,
    max_reactions: int = 12,
    max_genes: int = 8,
    max_mets: int = 6,
    max_pathways: int = 4,
) -> Dict[str, Any]:
    """
    Keeps met_json small + predictable for the model.
    """
    out: Dict[str, Any] = {}
    out["seeds"] = _coerce_list(met_json.get("seeds"))[:60]

    pvs = met_json.get("pathway_views") or []
    if isinstance(pvs, list):
        out["pathway_views"] = [
            {
                "label": str(p.get("label") or "Unknown"),
                "why_selected": str(p.get("why_selected") or ""),
                "reactions": _coerce_list(p.get("reactions"))[:30],
            }
            for p in pvs[:max_pathways]
        ]
    else:
        out["pathway_views"] = []

    rxs = met_json.get("reactions") or []
    trimmed = []
    if isinstance(rxs, list):
        for r in rxs[:max_reactions]:
            if not isinstance(r, dict):
                continue
            trimmed.append(
                {
                    "rid": r.get("rid"),
                    "irreversible": bool(r.get("irreversible")),
                    "compartment": r.get("compartment") or "NA",
                    "pathway": r.get("pathway") or "NA",
                    "genes": _coerce_list(r.get("genes"))[:max_genes],
                    # Prefer structured metabolites if present
                    "in_mets": (r.get("in_mets") or [])[:max_mets],
                    "out_mets": (r.get("out_mets") or [])[:max_mets],
                }
            )
    out["reactions"] = trimmed
    return out


def _normalize_result(
    raw: Dict[str, Any],
    *,
    base_answer: str,
) -> MetabolicRewriteResult:
    final_answer = (raw.get("final_answer") or raw.get("rewritten_answer") or "").strip()
    if not final_answer:
        # fallback: do not break the pipeline
        final_answer = base_answer.strip()

    # perspectives
    perspectives_raw = raw.get("perspectives") or []
    perspectives: List[MetabolicPerspective] = []
    if isinstance(perspectives_raw, list):
        for p in perspectives_raw:
            if not isinstance(p, dict):
                continue
            perspectives.append(
                MetabolicPerspective(
                    label=str(p.get("label") or "Unknown"),
                    mechanistic_story=str(p.get("mechanistic_story") or ""),
                    network_backed_steps=_coerce_list(p.get("network_backed_steps")),
                    evidence_genes=_coerce_list(p.get("evidence_genes")),
                    evidence_metabolites=_coerce_list(p.get("evidence_metabolites")),
                    evidence_reactions=_coerce_list(p.get("evidence_reactions")),
                    adds_beyond_literature=str(p.get("adds_beyond_literature") or ""),
                    limits_uncertainty=str(p.get("limits_uncertainty") or ""),
                )
            )

    # constraint findings
    findings_raw = raw.get("constraint_check") or raw.get("constraints") or []
    findings: List[ConstraintFinding] = []
    if isinstance(findings_raw, list):
        for f in findings_raw:
            if not isinstance(f, dict):
                continue
            findings.append(
                ConstraintFinding(
                    kind=str(f.get("kind") or "unknown"),
                    detail=str(f.get("detail") or ""),
                    related_reactions=_coerce_list(f.get("related_reactions")),
                )
            )

    metabolic_evidence_used = _coerce_list(raw.get("metabolic_evidence_used") or raw.get("evidence_used"))
    # keep BDRL-ish only, if model leaks other ids
    metabolic_evidence_used = [x for x in metabolic_evidence_used if str(x).startswith("BDRL")]

    warnings = _coerce_list(raw.get("warnings"))

    return MetabolicRewriteResult(
        final_answer=final_answer,
        perspectives=perspectives,
        constraint_check=findings,
        metabolic_evidence_used=metabolic_evidence_used,
        warnings=warnings,
        raw_json=raw,
    )





def build_met_blocks(met_json: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Returns:
      seed_block, pathway_block, reaction_block
    using your exact formatting style.
    """
    seed_list = _coerce_list(met_json.get("seeds"))
    seed_block = ", ".join(seed_list[:50]) if seed_list else "none"

    pathway_views = met_json.get("pathway_views", []) or []
    if isinstance(pathway_views, list):
        pathway_block = "\n".join(
            f"- {pv.get('label','Unknown')}: {pv.get('why_selected','')}"
            for pv in pathway_views[:4]
            if isinstance(pv, dict)
        ) or "none"
    else:
        pathway_block = "none"

    rxs = met_json.get("reactions", []) or []

    def rx_line(r: Dict[str, Any]) -> str:
        rid = r.get("rid", "NA")
        comp = r.get("compartment", "NA")
        irrev = "IRREV" if r.get("irreversible") else "REV/UNK"
        genes = ", ".join((r.get("genes") or [])[:8]) or "NA"

        in_mets = r.get("in_mets") or []
        out_mets = r.get("out_mets") or []

        def met_fmt(m: Dict[str, Any]) -> str:
            nm = m.get("name") or m.get("base_cid") or m.get("node")
            return f"{nm} ({m.get('node')})"

        ins = (
            ", ".join(met_fmt(m) for m in in_mets[:6])
            or ", ".join((r.get("in") or [])[:6])
            or "NA"
        )
        outs = (
            ", ".join(met_fmt(m) for m in out_mets[:6])
            or ", ".join((r.get("out") or [])[:6])
            or "NA"
        )

        pw = r.get("pathway") or "NA"
        return (
            f"* {rid} [{comp}] [{irrev}] pathway={pw}\n"
            f"  genes: {genes}\n"
            f"  in: {ins}\n"
            f"  out: {outs}"
        )

    reaction_block = (
        "\n".join(rx_line(r) for r in rxs[:12] if isinstance(r, dict))
        or "none"
    )

    return seed_block, pathway_block, reaction_block





# -------------------------
# Main entrypoint
# -------------------------

async def rewrite_answer_with_metabolic_grounding(
    *,
    question: str,
    base_answer: str,
    met_ctx: str,
    met_json: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_met_ctx_chars: int = 12000,
    max_met_ctx_lines: int = 180,
) -> MetabolicRewriteResult:
    """
    Second-pass rewriter:
    - Keeps base RAG answer as the main content
    - Adds 2–4 pathway perspectives grounded/validated by curated metabolic evidence
    - Emits strict JSON for downstream use

    Returns:
      MetabolicRewriteResult
    """
    if not question.strip():
        raise ValueError("question must be non-empty.")
    if not base_answer.strip():
        raise ValueError("base_answer must be non-empty.")

    key = _read_api_key_from_file_or_env(api_key=api_key)
    if not key:
        raise RuntimeError("OpenAI API key not found (env OPENAI_API_KEY or openai_key file).")

    met_ctx_short = _truncate_block(met_ctx, max_chars=max_met_ctx_chars, max_lines=max_met_ctx_lines)
    seed_block, pathway_block, reaction_block = build_met_blocks(met_json)
    

    system_prompt = (
        "You are a biomedical reasoning editor. "
        "You rewrite an existing answer into pathway-perspective structure while using "
        "a curated metabolic network as mechanistic grounding and a constraint validator. "
        "Additionally, you are also a knowledge reasoner who is also expertised in hypothesis reasoning, data explanation, hypothesis testing of disease and metabolism centric questions."
        "Do not hallucinate and go beyond evidences that are claimed"
        "Do not remove useful details from the base answer. "
        "Do not hallucinate network evidence: cite BDRL_id only when supported by the provided curated block."
    )

    user_prompt = f"""
        You will receive:
        (A) a question
        (B) a base answer from a literature RAG system (keep its wording, structure, and citations as much as possible)
        (C) a curated metabolic network report (met_ctx, human readable)
        (D) a curated metabolic network JSON slice (met_json, structured)

        GOAL:
        Rewrite (B) into a stronger response that keeps its content but adds:
        - 2–4 pathway-level perspectives while preserving the original answer's claims and biological story
        - Then enrich each perspective with mechanistic grounding using (C) and (D) where relevant.
        - An explicit constraint checking:
        * irreversibility: do not reason backwards for irreversible reactions
        * compartments: do not jump compartments without transport/explicit bridging in D
        - explicit labeling of claims NOT supported by curated network

        IMPORTANT BEHAVIOR:
        - Preserve the base answer’s overall content and citations.
        - Most importantly, you are a knowledge reasoner who is also expertised in hypothesis reasoning, data explanation, hypothesis testing of disease and metabolism centric questions.
        - The curated metabolic network is NOT the only evidence source.
        Use it to add mechanistic grounding and to validate mechanistic claims.
        - When you add a mechanistic step, cite a BDRL_id from (D). If you can’t, say “not supported by curated network”.
        - In the “Constraint check” section: List any violations found or state “No violations detected within provided curated reactions”.
        - In “Metabolic evidence actually used”, reproduce each used reaction exactly from section (D) “Top mechanistic reactions” including in/out, compartment, and IRREV label. If not present in (D), do not include it.
        OUTPUT: RETURN ONLY VALID JSON with this schema:
        {{
        "final_answer": "A well-formatted final answer text that includes sections: Base verbatim, Pathway perspectives, Constraint check, Metabolic evidence actually used",
        "perspectives": [
            {{
            "label": "Pathway label",
            "mechanistic_story": "write in plain English but adhering to the biological story and biochemical detail",
            "network_backed_steps": ["each bullet cites BDRL_id + compartment in text"],
            "evidence_genes": ["..."],
            "evidence_metabolites": ["Name (CID_comp)", "..."],
            "evidence_reactions": ["BDRL_M_..."],
            "adds_beyond_literature": "what network adds beyond literature-only reasoning",
            "limits_uncertainty": "limits/uncertainty of this perspective"
            }}
        ],
        "constraint_check": [
            {{
            "kind": "irreversibility|compartment|missing_transport|unsupported_claim|other",
            "detail": "clear explanation",
            "related_reactions": ["BDRL_M_..."] // list the full reactions
            }}
        ],
        "metabolic_evidence_used": ["BDRL_M_..."],  // only list what you truly used with full reactions
        "warnings": ["optional short warnings"]
        }}

        (A) QUESTION:
        {question}

        (B) BASE RAG ANSWER:
        <<<
        {base_answer}
        >>>

        (C) METABOLIC NETWORK REPORT:
        <<<
        {met_ctx_short}
        >>>

        (D) METABOLIC NETWORK EVIDENCE (human-readable blocks):
        Seeds:
        {seed_block}

        Pathway perspectives:
        {pathway_block}

        Top mechanistic reactions:
        {reaction_block}
        """

    client = AsyncOpenAI(base_url=base_url, api_key=key)
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    completion_text = completion.choices[0].message.content or ""
    parsed = _convert_response_to_json(completion_text)
    return _normalize_result(parsed, base_answer=base_answer)
