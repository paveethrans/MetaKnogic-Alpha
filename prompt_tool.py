"""Utilities for rewriting user prompts before sending them to the RAG stack.

The main entrypoint is `rewrite_prompt_with_metabolism_signal`, which uses an
LLM to:
1) rewrite the user prompt for clarity and detail,
2) highlight intent and missing details, and
3) emit a boolean signal indicating whether the query contains metabolic /
   biochemical content so downstream code can trigger graph retrieval.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path
from openai import AsyncOpenAI  # type: ignore

DEFAULT_MODEL = "gpt-5.1"
key_path = Path(__file__).with_name("openai_key")
if not key_path.exists():
    raise RuntimeError("openai_key file not found next to app.py")
os.environ["OPENAI_API_KEY"] = key_path.read_text(encoding="utf-8").strip()


logger = logging.getLogger(__name__)


@dataclass
class PromptRewriteResult:
    """Structured output for prompt rewriting."""

    rewritten_prompt: str
    intent_summary: str
    metabolic_signal: bool
    metabolic_terms: List[str]
    reaction_clues: List[str]
    missing_details: List[str]
    entities_block: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_list(value: Any) -> List[str]:
    """Ensure list-of-str output."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value).strip()] if str(value).strip() else []


def _normalize_result(raw: Dict[str, Any], original_prompt: str) -> PromptRewriteResult:
    """Fill defaults and type-normalize the model output."""
    rewritten_prompt = raw.get("rewritten_prompt") or original_prompt
    intent_summary = raw.get("intent_summary") or raw.get("intent") or ""
    metabolic_signal = bool(raw.get("metabolic_signal") or raw.get("metabolite_signal"))
    metabolic_terms = _coerce_list(
        raw.get("metabolic_terms") or raw.get("metabolites") or raw.get("entities")
    )
    reaction_clues = _coerce_list(
        raw.get("reaction_clues") or raw.get("reactions") or raw.get("pathways")
    )
    missing_details = _coerce_list(
        raw.get("missing_details") or raw.get("gaps") or raw.get("clarifications")
    )

    # If the model surfaced metabolic hints, ensure the signal is set.
    if not metabolic_signal and (metabolic_terms or reaction_clues):
        metabolic_signal = True

    entities_block = str(raw.get("entities_block") or "").strip()

    return PromptRewriteResult(
        rewritten_prompt=rewritten_prompt.strip(),
        intent_summary=intent_summary.strip(),
        metabolic_signal=metabolic_signal,
        metabolic_terms=metabolic_terms,
        reaction_clues=reaction_clues,
        missing_details=missing_details,
        entities_block=entities_block,
    )


def _convert_response_to_json(response: str) -> Dict[str, Any]:
    """Parse LLM output into JSON; tolerate leading/trailing text."""
    if not isinstance(response, str):
        raise ValueError("LLM response is not a string.")
    response = response.strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to locate first JSON object in the text
    match = re.search(r"{.*}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError("Unable to parse JSON from LLM response.")


async def rewrite_prompt_with_metabolism_signal(
    original_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    extra_context: Optional[str] = None,
) -> PromptRewriteResult:
    """Rewrite a user prompt and emit a metabolic-content signal.

    Args:
        original_prompt: The raw user prompt.
        model: Chat model to use (defaults to GPT-4o-mini).
        base_url: Optional override for OpenAI-compatible endpoints.
        api_key: Optional API key.
        temperature: Sampling temperature for the rewrite call.
        extra_context: Optional extra domain hints to bias the rewrite.

    Returns:
        PromptRewriteResult containing the rewritten prompt, detected intent,
        a metabolic flag, and extracted metabolic/reaction clues.
    """
    if not original_prompt or not original_prompt.strip():
        raise ValueError("original_prompt must be a non-empty string.")

    system_prompt = (
        "You are a domain-aware prompt optimizer for biomedical and metabolic QA. "
        "Rewrite user prompts to clarify intent, add helpful biochemical detail "
        "when hinted (metabolites, reactions, pathways, enzymes, genes), and flag "
        "whether downstream metabolic graph lookup should be triggered. "
        "Be concise and avoid hallucinating unsupported facts."
    )

    instructions = f"""
        Rewrite the user's prompt to maximize retrieval and keep the user intent explicit.
        Even if the user does not realize their question is metabolism-related, check for
        implicit metabolic/biochemical cues (metabolites, enzymes, genes, pathways,
        biomarkers, symptoms that imply pathway involvement, reaction conditions, kinetics).
        If any such cues are present or implied, incorporate them in the rewrite while
        staying faithful to the user's ask.

        Original prompt: {original_prompt.strip()}
        Extra context (optional): {extra_context.strip() if extra_context else "N/A"}

        Return ONLY valid JSON with this shape:
        {{
        "rewritten_prompt": "rewritten prompt with clarified details",
        "intent_summary": "1-2 sentence summary of what the user wants",
        "metabolic_signal": true | false,  // true if metabolic/biochemical content is present or implied
        "metabolic_terms": ["list", "of", "metabolites/enzymes/genes/pathways"],
        "reaction_clues": ["specific full-reactions, pathways, or processes to check in the metabolic graph"],
        "missing_details": ["key gaps to ask the user for if clarification is needed"]
        "entities_block": "GENE: ...\nMET: ...\n..."
        }}

        Additionally, must include a field "entities_block" as a SINGLE STRING with one entity per line,
        using ONLY these line formats:

        GENE: <HGNC_SYMBOL>
        MET: <canonical metabolite name>
        MET_SYNONYM: <common synonym or abbreviation>
        PATHWAY: <pathway name>
        PROCESS: <process term>

        Rules for entities_block:
        - Include genes, metabolites whenever present or strongly implied.
        - For each MET line, add 2-4 MET_SYNONYM lines if common (e.g., "α-ketoglutarate", "alpha ketoglutarate", "2-OG").
        - Do not invent new entities. If unsure, omit.
        - Keep names as clean standalone strings and in their original written way (no extra punctuation).

        Update the JSON shape by adding:
        "entities_block": "GENE: ...\nMET: ...\n..."

        Return ONLY valid JSON (no extra text).

        Rules:
        - Keep the rewritten prompt grounded in the user's wording; do not invent facts.
        - When metabolic_signal is true, prefer explicit metabolite/reaction hints if present.
        - If no metabolic cues exist, set metabolic_signal to false and leave lists empty.
        - Keep outputs in the same language as the input unless the user explicitly asks otherwise.
        """

    logger.debug("Rewriting prompt with model=%s, base_url=%s", model, base_url)

    try:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instructions},
            ],
            temperature=temperature,
        )
        completion_text = completion.choices[0].message.content
    except Exception as exc:
        logger.error("Prompt rewrite failed: %s", exc)
        raise

    try:
        parsed = _convert_response_to_json(completion_text)
    except Exception as exc:
        logger.error("Failed to parse rewrite response: %s", exc)
        raise

    if not isinstance(parsed, dict):
        logger.error("Unexpected rewrite response type: %s", type(parsed))
        raise ValueError("LLM response is not a JSON object.")

    result = _normalize_result(parsed, original_prompt=original_prompt)
    logger.debug(
        "Rewrite result: signal=%s, terms=%s, reactions=%s",
        result.metabolic_signal,
        result.metabolic_terms,
        result.reaction_clues,
    )
    return result