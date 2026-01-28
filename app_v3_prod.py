# app.py

import os
import sys, json
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List
import re
import glob
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils_for_graph_v2 import build_subgraph_for_query
from hypergraphrag import HyperGraphRAG
from hypergraphrag.base import QueryParam
from hypergraphrag.llm import openai_complete
from sentence_emb_model import embed_batch
from hypergraphrag.utils import set_logger, logger
from prompt_tool import rewrite_prompt_with_metabolism_signal

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static_v3", StaticFiles(directory="static_v3"), name="static_v3")


# ----------------------------
# Paths (your environment) CHANGE FOR EC2
# ----------------------------
SCCHAT_ROOT = Path("/home/ubuntu/data/projects/WebDemoKHGRAG/scChat").resolve()
HYPERGRAPH_ROOT = Path("/home/ubuntu/data/projects/WebDemoKHGRAG/").resolve()

for p in [str(SCCHAT_ROOT), str(HYPERGRAPH_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Remember the directory where app.py is running from
APP_CWD = Path.cwd()

# Absolute HyperGraphRAG working directory (fixes FileNotFoundError when scChat changes cwd)
RAG_WORKDIR = (HYPERGRAPH_ROOT / "expr" / "pmc_neo4j_milvus").resolve()
RAG_WORKDIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# scChat config (keep your behavior)
# ----------------------------
os.environ.setdefault("SCCHAT_USE_SCVI", "0")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "scchatbot.settings")

# This is the key: tell scChat to skip startup annotation
# (Your scChat code must actually respect this env var; see note below.)
os.environ.setdefault("SCCHAT_RUN_STARTUP_ANNOTATION", "0")

# Optional: fix relative config paths inside scChat
# If scChat reads "media/specification_graph.json" as a relative path,
# forcing cwd to SCCHAT_ROOT makes it resolvable.
os.environ.setdefault("SCCHAT_FORCE_CWD", "1")


# ----------------------------
# Global single-worker lock (YOU said 1 worker is enough)
# This serializes the whole /api/query_with_file pipeline.
# ----------------------------
PIPELINE_LOCK = asyncio.Lock()

# Optional: avoid hanging forever
SCCHAT_TIMEOUT_SEC = int(os.environ.get("SCCHAT_TIMEOUT_SEC", "7200"))  # 2 hours default
LOCK_WAIT_TIMEOUT_SEC = int(os.environ.get("LOCK_WAIT_TIMEOUT_SEC", "10800")) # 3 hours default


def load_mdHash_2_entityName(file_path: str) -> dict[str, str]:
    with open(file_path, "r") as f:
        return json.load(f)

async def prompt_rewrite(original_prompt: str):
    """Call the rewrite helper with a sample prompt and print the output."""
    result = await rewrite_prompt_with_metabolism_signal(original_prompt)
    return result


async def modernpubmed_embedding(texts: list[str]):
    return embed_batch(texts, batch_size=16, normalize=True, max_seq_length=2048)

modernpubmed_embedding.embedding_dim = 768

# ----------------------------
# Databases / env vars
# ----------------------------

os.environ["HGRAG_NEO4J_STATIC"] = "1"

os.environ["NEO4J_URI"] = "bolt://localhost:7688"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "myneo4jgraph"

os.environ["MILVUS_URI"] = "http://localhost:19530"
os.environ["MILVUS_USER"] = "root"
os.environ["MILVUS_PASSWORD"] = "Milvus"
os.environ["MILVUS_DB_NAME"] = "hypergraph_pmc_static_fast"


# ----------------------------
# LLM key
# ----------------------------
key_path = Path(__file__).with_name("openai_key")
if not key_path.exists():
    raise RuntimeError("openai_key file not found next to app.py")
os.environ["OPENAI_API_KEY"] = key_path.read_text(encoding="utf-8").strip()


# ----------------------------
# Init HyperGraphRAG
# ----------------------------
"""
rag = HyperGraphRAG(
    working_dir=str(RAG_WORKDIR),
    # working_dir="expr/example",
    # vector_storage="MilvusVectorDBStorge",
    # graph_storage="Neo4JStorage",
    # embedding_func=modernpubmed_embedding,
    # llm_model_func=openai_complete,
    # llm_model_name="gpt-5.1",
)
"""

# You already know this works for one query; we just make it robust for many
rag = HyperGraphRAG(
    working_dir="expr/pmc_neo4j_milvus",
    vector_storage="MilvusVectorDBStorge",
    graph_storage="Neo4JStorage",
    embedding_func=modernpubmed_embedding,
    llm_model_func=openai_complete,
    llm_model_name="gpt-5.1",
    addon_params={"example_number": 1},  # keep keyword-extraction prompt compact
)

mdHash_2_entityName = load_mdHash_2_entityName("expr/pmc_neo4j_milvus/mdHash_2_entityName.json")

# ----------------------------
# Mount scChat Django ASGI under /scchat (optional)
# ----------------------------
try:
    from django.core.asgi import get_asgi_application

    django_asgi_app = get_asgi_application()
    app.mount("/scchat", django_asgi_app)
    print("\n\n######## Mounted scChat Django ASGI app at /scchat ########\n\n")
except Exception as e:
    print("WARNING: Failed to mount scChat Django ASGI app:", repr(e))
    django_asgi_app = None



# ----------------------------
# Upload store
# ----------------------------
SCCHAT_MEDIA_DIR = (SCCHAT_ROOT / "media").resolve()
# UPLOAD_DIR = Path("./uploads").resolve()
SCCHAT_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# optional: keep per-session mapping paths here
SESSION_MAPPINGS: Dict[str, str] = {}

# session_id -> file path
SESSION_FILES: Dict[str, str] = {}

# Lazy-loaded scChat bot
_SCCHAT_BOT = None



def get_scchat_bot():
    """
    Lazy import so scChat doesn’t initialize too early.
    IMPORTANT:
      This assumes your chatbot class supports:
        - attach_dataset(session_id, file_path)
        - send_message(message, session_id=...)
    """
    global _SCCHAT_BOT

    # Force cwd to SCCHAT_ROOT if you want scChat to resolve relative paths
    if os.environ.get("SCCHAT_FORCE_CWD", "0") == "1":
        os.chdir(str(SCCHAT_ROOT))


    if _SCCHAT_BOT is None:
        # Prefer your patched class; fallback if you kept the old name.
        try:
            from scchatbot.chatbot import MultiAgentChatBot  # your patched bot
            _SCCHAT_BOT = MultiAgentChatBot()
        except Exception:
            from scchatbot.chatbot import ChatBot  # upstream
            _SCCHAT_BOT = ChatBot()

    
    return _SCCHAT_BOT




# ----------------------------

# Helper functions for auto creating sample mapping file

# ----------------------------
def _safe_json_dump(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4), encoding="utf-8")


def _pick_sample_column_from_obs(obs_df) -> Optional[str]:
    preferred = [
        "sample", "sample_id", "sampleid", "sample_name", "samplename",
        "orig.ident", "orig_ident", "origident",
        "batch", "batch_id", "batchid",
        "library", "library_id",
        "donor", "donor_id",
        "patient", "patient_id",
        "condition", "treatment", "timepoint",
        "group",
    ]

    cols = list(obs_df.columns)
    cols_lower = {c.lower(): c for c in cols}  # map lowercase -> actual name

    # quick pass on preferred names (case-insensitive)
    for c in preferred:
        actual = cols_lower.get(c.lower())
        if actual is None:
            continue

        nunique = obs_df[actual].nunique(dropna=True)
        n = len(obs_df)
        if nunique > 1 and nunique < max(50, int(0.2 * n)):
            return actual

    # broader scoring pass
    best = None
    best_score = -1

    n = len(obs_df)

    for c in cols:
        s = obs_df[c]
        # skip numeric columns usually not sample labels
        if str(s.dtype).startswith(("int", "float")):
            continue

        nunique = s.nunique(dropna=True)
        if nunique <= 1:
            continue

        # exclude cell-barcode-like columns (almost all unique)
        if nunique > max(200, int(0.5 * n)):
            continue

        # score: fewer unique (but not 1) is better, categorical is better, short name match is better
        score = 0
        name_l = c.lower()
        if any(k in name_l for k in ["sample", "orig", "batch", "patient", "donor", "condition", "treat", "time", "group"]):
            score += 3
        if str(s.dtype) == "category":
            score += 2
        # prefer "reasonable" unique count
        if nunique <= 50:
            score += 2
        if nunique <= 20:
            score += 1

        if score > best_score:
            best = c
            best_score = score

    return best


def _infer_description(sample_label: str) -> str:
    """
    Build a decent English description from a sample label like:
      patient1_IP
      patient3_Peak
      P2_pre
      donor7_treated
    """
    raw = str(sample_label)

    # normalize separators to spaces for parsing
    norm = re.sub(r"[\s\-]+", "_", raw.strip())
    parts = [p for p in norm.split("_") if p]

    # patient/donor detection
    patient_num = None
    donor_num = None
    phase = None

    # common phase synonyms
    phase_map = {
        "ip": "IP phase",
        "peak": "Peak phase",
        "pre": "pre-treatment",
        "post": "post-treatment",
        "ctrl": "control",
        "control": "control",
        "treated": "treated",
        "treat": "treated",
        "baseline": "baseline",
        "followup": "follow-up",
        "fu": "follow-up",
    }

    # find patient/donor tokens
    for p in parts:
        m = re.match(r"(?i)patient(\d+)$", p)
        if m:
            patient_num = m.group(1)
            continue
        m = re.match(r"(?i)p(\d+)$", p)
        if m and patient_num is None:
            patient_num = m.group(1)
            continue
        m = re.match(r"(?i)donor(\d+)$", p)
        if m:
            donor_num = m.group(1)
            continue
        m = re.match(r"(?i)d(\d+)$", p)
        if m and donor_num is None:
            donor_num = m.group(1)
            continue

        key = p.lower()
        if key in phase_map and phase is None:
            phase = phase_map[key]

    # build description
    if patient_num and phase:
        return f"This sample is from patient {patient_num} during the {phase}."
    if donor_num and phase:
        return f"This sample is from donor {donor_num} during the {phase}."
    if patient_num:
        return f"This sample is from patient {patient_num}."
    if donor_num:
        return f"This sample is from donor {donor_num}."
    if phase:
        return f"This sample corresponds to the {phase} condition."
    return f"This sample corresponds to '{raw}'."

def _auto_create_sample_mapping_json(
    sid: str,
    h5ad_path: str,
    media_dir: Path,
    *,
    mapping_filename: Optional[str] = None,
) -> str:
    try:
        import anndata as ad
    except Exception:
        import scanpy as sc
        ad = None

    h5ad_p = Path(h5ad_path).resolve()
    if not h5ad_p.exists():
        raise FileNotFoundError(f"h5ad not found: {h5ad_p}")

    if mapping_filename is None:
        mapping_filename = "sample_mapping.json"

    out_path = (media_dir / mapping_filename).resolve()
    media_dir.mkdir(parents=True, exist_ok=True)

    # remove old mapping if it exists
    if out_path.exists():
        out_path.unlink()

    if ad is not None:
        adata = ad.read_h5ad(str(h5ad_p), backed="r")
    else:
        adata = sc.read_h5ad(str(h5ad_p), backed="r")

    obs = adata.obs
    sample_col = _pick_sample_column_from_obs(obs)

    # fallback: single "overall"
    if sample_col is None:
        mapping = {
            "Sample name": None,
            "Sample categories": {"overall": "overall"},
            "Sample description": {
                "overall": f"Auto-created mapping on {datetime.now().isoformat(timespec='seconds')} (no sample column detected)."
            },
            "_meta": {
                "session_id": sid,
                "h5ad": str(h5ad_p),
                "chosen_obs_column": None,
                "num_samples": 1,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        }
        _safe_json_dump(out_path, mapping)
        return str(out_path)

    values = obs[sample_col].astype(str).fillna("overall").tolist()
    uniq = sorted(set(v for v in values if v.strip()))

    categories = {u: u for u in uniq}
    descriptions = {u: _infer_description(u) for u in uniq}

    mapping = {
        "Sample name": sample_col,  # ✅ critical fix
        "Sample categories": categories,
        "Sample description": descriptions,
        "_meta": {
            "session_id": sid,
            "h5ad": str(h5ad_p),
            "chosen_obs_column": sample_col,
            "num_samples": len(uniq),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    }

    print("################\nAuto-created sample mapping:", mapping, "\n################")
    # _safe_json_dump(out_path, mapping)
    return str(out_path)






def clean_answer(raw: str) -> str:
    if raw is None:
        return ""
    lines = []
    for line in str(raw).splitlines():
        if line.strip().startswith("INFO:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _new_sid() -> str:
    return str(uuid.uuid4())[:8]


def _save_upload_for_session(sid: str, file: UploadFile) -> str:
    safe_name = Path(file.filename).name
    print('Original filename:', file.filename, 'Safe name:', safe_name)

    # delete any existing h5ad for this session (uploads + scChat media)
    for p in [
        SCCHAT_MEDIA_DIR / f"*.h5ad"
    ]:
        for old in glob.glob(str(p)):
            try:
                os.remove(old)
                print("Deleted old:", old)
            except Exception as e:
                print("Could not delete:", old, e)

    target_path = SCCHAT_MEDIA_DIR / f"{sid}__{safe_name}"
    with target_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    SESSION_FILES[sid] = str(target_path)
    print(f"Saved upload for session {sid} to {target_path}")
    return str(target_path)


def _require_h5ad(file_path: str):
    # If you truly want to accept other formats later, relax this check.
    if not file_path.lower().endswith(".h5ad"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded dataset must be a .h5ad file for scChat. Please upload a .h5ad.",
        )




def _wrap_plot_html(plot_html: str) -> str:
    """
    Ensures Plotly exists for the plot HTML.
    If the snippet already contains <html> or plotly script, we leave it alone.
    """
    if plot_html is None:
        return ""

    s = str(plot_html)

    # If it's already a full HTML doc, return as-is
    if "<html" in s.lower():
        return s

    # If it already loads plotly, return as-is
    if "plotly" in s.lower():
        return s

    # Otherwise wrap with a minimal HTML shell + plotly CDN
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://cdn.plot.ly/plotly-2.34.0.min.js"></script>
  <style>
    html, body {{ margin:0; padding:0; width:100%; height:100%; background:white; }}
  </style>
</head>
<body>
{s}
</body>
</html>
"""


def parse_scchat_response(raw: str):
    """
    Tries to parse scChat output as JSON so we can extract plots.
    Returns: (answer_text, plots_list)
    plots_list is List[PlotItem]
    """
    if raw is None:
        return "", []

    # First attempt: JSON parse
    try:
        obj = json.loads(raw)
        # common cases:
        # - {"response": "...", "plots": [{"title": "...", "html": "..."}]}
        # - {"messages": [{"type":"plot","title":...,"html":...}, {"type":"text","text":...}]}
        plots = []
        answer_text = ""

        if isinstance(obj, dict):
            # Case 1: "plots" array
            if "plots" in obj and isinstance(obj["plots"], list):
                for p in obj["plots"]:
                    if isinstance(p, dict) and "html" in p:
                        title = str(p.get("title", "Plot"))
                        html = _wrap_plot_html(p.get("html", ""))
                        plots.append(PlotItem(title=title, html=html))

            # Case 2: "messages" array
            if "messages" in obj and isinstance(obj["messages"], list):
                for m in obj["messages"]:
                    if not isinstance(m, dict):
                        continue
                    mtype = str(m.get("type", "")).lower()
                    if mtype == "plot" and "html" in m:
                        title = str(m.get("title", "Plot"))
                        html = _wrap_plot_html(m.get("html", ""))
                        plots.append(PlotItem(title=title, html=html))
                    elif mtype in ("text", "answer", "response") and ("text" in m or "response" in m):
                        answer_text += (m.get("text") or m.get("response") or "")

            # Case 3: "response" or "answer" text
            if not answer_text:
                if isinstance(obj.get("response"), str):
                    answer_text = obj["response"]
                elif isinstance(obj.get("answer"), str):
                    answer_text = obj["answer"]

        # Fallback if JSON had no clear answer
        if not answer_text:
            answer_text = raw

        return clean_answer(answer_text), plots

    except Exception:
        # Not JSON: treat as normal text
        return clean_answer(raw), []






# ----------------------------
# API Models
# ----------------------------
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class PlotItem(BaseModel):
    title: str
    html: str


class QueryResponse(BaseModel):
    answer: str
    graph: Optional[Dict[str, Any]] = None
    routed_to: Literal["kg", "hybrid"]
    session_id: str
    plots: Optional[List[PlotItem]] = None


class AttachRequest(BaseModel):
    session_id: str
    stored_path: str  # path returned by /api/upload (or your own path)


# ----------------------------
# Pages
# ----------------------------
@app.get("/about", response_class=HTMLResponse)
def about_page():
    return Path("static_v3/about_gemini.html").read_text(encoding="utf-8")


@app.get("/team", response_class=HTMLResponse)
def team_page():
    return Path("static_v3/team.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def landing_page():
    return Path("static_v3/index_gemini.html").read_text(encoding="utf-8")


# ----------------------------
# Upload endpoint
# ----------------------------
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    sid = session_id or _new_sid()
    stored_path = _save_upload_for_session(sid, file)

    return {
        "status": "success",
        "session_id": sid,
        "filename": Path(file.filename).name,
        "stored_path": stored_path,
    }


# ----------------------------
# Optional: attach an already uploaded file to scChat (no query)
# Use this if you want a 2-step flow: upload -> attach -> chat
# ----------------------------
# @app.post("/api/attach")
# async def attach_dataset(payload: AttachRequest):
#     sid = (payload.session_id or "").strip()
#     if not sid:
#         raise HTTPException(status_code=400, detail="session_id is required")

#     file_path = (payload.stored_path or "").strip()
#     if not file_path:
#         raise HTTPException(status_code=400, detail="stored_path is required")

#     if not Path(file_path).exists():
#         raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

#     _require_h5ad(file_path)

#     # Auto-create a per-session mapping and store in scChat media
#     mapping_path = await asyncio.to_thread(
#         _auto_create_sample_mapping_json,
#         sid,
#         file_path,
#         SCCHAT_MEDIA_DIR,
#         mapping_filename="sample_mapping.json",
#     )
#     SESSION_MAPPINGS[sid] = mapping_path

#     # If your scChat code can read this env var, set it before attach/send.
#     os.environ["SCCHAT_SAMPLE_MAPPING_PATH"] = mapping_path

#     bot = get_scchat_bot()

#     # Attach can be heavy; run in thread.
#     try:
#         await asyncio.wait_for(
#             asyncio.to_thread(bot.attach_dataset, sid, file_path),
#             timeout=SCCHAT_TIMEOUT_SEC,
#         )
#     except asyncio.TimeoutError:
#         raise HTTPException(status_code=504, detail=f"attach_dataset timed out after {SCCHAT_TIMEOUT_SEC}s")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"attach_dataset failed: {e}")

#     # Remember mapping so /api/query_with_file can skip re-upload
#     SESSION_FILES[sid] = file_path

#     return {"status": "success", "session_id": sid, "attached_path": file_path}


# ----------------------------
# Main KG-only query endpoint (no scChat)
# ----------------------------
@app.post("/api/query", response_model=QueryResponse)
async def run_query(payload: QueryRequest):
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    logger.info("Received query: %s", query)

    rewritten_result = await prompt_rewrite(query)
    rewritten_query = rewritten_result.rewritten_prompt
    logger.info("Rewritten question: %s", rewritten_query)

    sid = payload.session_id or _new_sid()

    os.chdir(str(APP_CWD))
    # Reduce retrieved items and context sizes to avoid token overflow
    param = QueryParam(
        top_k=15,
        max_token_for_text_unit=13000,
        max_token_for_global_context=2000,
        max_token_for_local_context=2000,
    )
    result = await rag.aquery(rewritten_query, param)
    if isinstance(result, str):
        answer_text = result
    elif isinstance(result, dict):
        answer_text = result.get("answer", str(result))
    else:
        answer_text = str(result)

    answer_text = clean_answer(answer_text)

    graph_json = await build_subgraph_for_query(
        rag=rag,
        query_text=rewritten_query,
        top_k=10,
        hops=2,
        max_entities=50,
        max_hyperedges=50,
        mdHash_2_entityName=mdHash_2_entityName,
    )

    answer_text = f"## Rewritten question:\n\n{rewritten_query}\n\n## Answer:\n{answer_text}"

    return QueryResponse(
        answer=answer_text,
        graph=graph_json,
        routed_to="kg",
        session_id=sid,
        plots=[],
    )


# ----------------------------
# Hybrid endpoint: upload + scChat + HyperGraphRAG
# Single-worker serialized by PIPELINE_LOCK for one user at a time processing.
# ----------------------------
@app.post("/api/query_with_file", response_model=QueryResponse)
async def query_with_file(
    file: UploadFile = File(...),
    query: str = Form(...),
    session_id: Optional[str] = Form(None),
):
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    sid = session_id or _new_sid()

    # Save/overwrite the session file
    file_path = _save_upload_for_session(sid, file)
    _require_h5ad(file_path)

    # Auto-create mapping for this uploaded h5ad
    mapping_path = await asyncio.to_thread(
        _auto_create_sample_mapping_json,
        sid,
        file_path,
        SCCHAT_MEDIA_DIR,
        mapping_filename="sample_mapping.json",
    )
    SESSION_MAPPINGS[sid] = mapping_path
    os.environ["SCCHAT_SAMPLE_MAPPING_PATH"] = mapping_path

    # Global single-worker serialization
    try:
        await asyncio.wait_for(PIPELINE_LOCK.acquire(), timeout=LOCK_WAIT_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=429, detail="Pipeline is busy. Please try again later.")

    try:
        os.chdir(str(SCCHAT_ROOT))
        os.environ.setdefault("SCCHAT_RUN_STARTUP_ANNOTATION", "1")
        bot = get_scchat_bot()
        print('\n\n Getting bot and starting pipeline for session', sid)

        # # 1) Attach only if this session doesn't have an attached dataset OR it changed
        if SESSION_FILES.get(sid) != file_path:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(bot.attach_dataset, sid, file_path),
                    timeout=SCCHAT_TIMEOUT_SEC,
                )
                print('\n\n Attach dataset completed')
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail=f"attach_dataset timed out after {SCCHAT_TIMEOUT_SEC}s")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"attach_dataset failed: {e}")

        # 2) scChat message (threaded)
        user_message = (
            "A dataset has been attached for this session.\n"
            f"Dataset path: {file_path}\n\n"
            f"User question:\n{q}\n\n"
            "Instructions:\n"
            "Use the analysis from the uploaded file as the primary evidence source when relevant. And use this to generate a detailed answer to the user's question. "
            "If the dataset does not contain sufficient information, say so explicitly."
        )

        try:
            sc_chat_response_text = await asyncio.wait_for(
                asyncio.to_thread(bot.send_message, user_message, sid),
                timeout=SCCHAT_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"scChat timed out after {SCCHAT_TIMEOUT_SEC}s (session={sid}).")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"scChat route failed: {e}")

        sc_chat_answer, scchat_plots = parse_scchat_response(sc_chat_response_text)
        print('\n\n scChat message completed', sc_chat_answer)

        # 3) Feed into HyperGraphRAG
        # Important: restore cwd so HyperGraphRAG relative paths don't break later
        os.chdir(str(APP_CWD))
        final_query = (
            q
            + "\n\nThis is additional context extracted from a user-provided data file. Use the analysis as the primary evidence source when relevant. And use this to generate a detailed, meaningful and highly relevant answer to the user's question:\n"
            + sc_chat_answer
        )

        print('\n\n Final query to HyperGraphRAG:', final_query[:200])

        result = await rag.aquery(final_query)
        if isinstance(result, str):
            answer_text = result
        elif isinstance(result, dict):
            answer_text = result.get("answer", str(result))
        else:
            answer_text = str(result)

        answer_text = clean_answer(answer_text)

        graph_json = await build_subgraph_for_query(
            rag=rag,
            query_text=final_query,
            top_k=10,
            hops=4,
            max_entities=50,
            max_hyperedges=50,
        )

        return QueryResponse(
            answer=answer_text,
            graph=graph_json,
            routed_to="hybrid",
            session_id=sid,
            plots=scchat_plots,
        )

    finally:
        PIPELINE_LOCK.release()