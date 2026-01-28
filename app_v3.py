# app.py

import os
import sys, json
import uuid
import shutil
import asyncio
import contextlib
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List
import re
import glob
from datetime import datetime
import time

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from metabolite_network_integration.metabolic_runtime import MetabolicNetwork

from utils_for_graph_v2 import build_subgraph_for_query
from hypergraphrag.base import QueryParam
from hypergraphrag.llm import openai_complete
from hypergraphrag.utils import set_logger, logger
from prompt_tool import rewrite_prompt_with_metabolism_signal
from metabolic_prompt_rewritting import rewrite_answer_with_metabolic_grounding
from hypergraphrag import HyperGraphRAG

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
SCCHAT_ROOT = Path("./scChat").resolve()
print(SCCHAT_ROOT)
HYPERGRAPH_ROOT = Path(".").resolve()

for p in [str(SCCHAT_ROOT), str(HYPERGRAPH_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Remember the directory where app.py is running from
APP_CWD = Path.cwd()

# Absolute HyperGraphRAG working directory (fixes FileNotFoundError when scChat changes cwd)
RAG_WORKDIR = (HYPERGRAPH_ROOT / "expr" / "example").resolve()
RAG_WORKDIR.mkdir(parents=True, exist_ok=True)

SCCHAT_SESSION_BASEDIR = (SCCHAT_ROOT / "SESSION").resolve()
SCCHAT_SESSION_BASEDIR.mkdir(parents=True, exist_ok=True)



async def prompt_rewrite(original_prompt: str):
    """Call the rewrite helper with a sample prompt and print the output."""
    result = await rewrite_prompt_with_metabolism_signal(original_prompt)
    return result



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


# ----------------------------
# Databases / env vars
# ----------------------------
os.environ["HGRAG_NEO4J_STATIC"] = "1"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
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
rag = HyperGraphRAG(
    working_dir=str(RAG_WORKDIR),
    # working_dir="expr/example",
    # vector_storage="MilvusVectorDBStorge",
    # graph_storage="Neo4JStorage",
    # embedding_func=modernpubmed_embedding,
    # llm_model_func=openai_complete,
    # llm_model_name="gpt-5.1",
)


MET = MetabolicNetwork(
    graph_path="metabolite_network_integration/metabolic_graph_outputs/met_graph.pkl",
    name_id_metabolite_mapping="metabolite_network_integration/metabolic_graph_outputs/name_id_met_mapping.json",
    gene_symbols_path="metabolite_network_integration/metabolic_graph_outputs/gene_symbols.json",
    mode=os.environ.get("MET_RETRIEVER_MODE", "ppr"),  # set to "ppr" if you want
)


# ----------------------------
# Mount scChat Django ASGI under /scchat (optional)
# ----------------------------
# try:
#     from django.core.asgi import get_asgi_application

#     django_asgi_app = get_asgi_application()
#     app.mount("/scchat", django_asgi_app)
#     print("\n\n######## Mounted scChat Django ASGI app at /scchat ########\n\n")
# except Exception as e:
#     print("WARNING: Failed to mount scChat Django ASGI app:", repr(e))
#     django_asgi_app = None


# ----------------------------
# Upload store
# ----------------------------
SCCHAT_SHARED_MEDIA_DIR = (SCCHAT_ROOT / "media").resolve()
SCCHAT_SHARED_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# optional: keep per-session mapping paths here
SESSION_MAPPINGS: Dict[str, str] = {}

# session_id -> file path
SESSION_FILES: Dict[str, str] = {}

# Per-session scChat bots (in-memory)
_SCCHAT_BOTS: Dict[str, Any] = {}
_SCCHAT_BOT_META: Dict[str, Dict[str, Any]] = {}
SCCHAT_BOT_TTL_SEC = int(os.environ.get("SCCHAT_BOT_TTL_SEC", "604800"))  # 1 week of inactivity of a bot


@contextlib.contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))


def _session_dir(sid: str) -> Path:
    return (SCCHAT_SESSION_BASEDIR / sid).resolve()


def _session_media_dir(sid: str) -> Path:
    return (_session_dir(sid) / "media").resolve()

def _sanitize_path_segment(value: Optional[str], fallback: str) -> str:
    s = (value or "").strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-._")
    if not s:
        return fallback
    return s[:80]


def _inquiry_dir(sid: str, inquiry_id: str) -> Path:
    safe_inquiry = _sanitize_path_segment(inquiry_id, "inquiry")
    return (_session_dir(sid) / "inquiries" / safe_inquiry).resolve()


def _ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    base = path
    for i in range(1, 10_000):
        cand = base.with_name(f"{base.name}-{i}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find unique directory name for {base}")


def _file_fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _cleanup_expired_scchat_sessions(now: Optional[float] = None):
    now = time.time() if now is None else now
    expired = []
    for sid, meta in list(_SCCHAT_BOT_META.items()):
        last_access = float(meta.get("last_access", 0))
        if now - last_access > SCCHAT_BOT_TTL_SEC:
            expired.append(sid)

    for sid in expired:
        bot = _SCCHAT_BOTS.pop(sid, None)
        _SCCHAT_BOT_META.pop(sid, None)
        try:
            if bot is not None and hasattr(bot, "cleanup"):
                bot.cleanup()
        except Exception:
            pass


def _get_or_create_scchat_bot(sid: str, dataset_path: str):
    """
    Returns a per-session scChat bot instance pinned to a per-session workspace.
    If the dataset changes for the same session, the bot is recreated.
    """
    dataset_p = Path(dataset_path).resolve()
    fp = _file_fingerprint(dataset_p)

    existing = _SCCHAT_BOTS.get(sid)
    meta = _SCCHAT_BOT_META.get(sid) or {}
    if existing is not None and meta.get("dataset_fingerprint") == fp:
        meta["last_access"] = time.time()
        _SCCHAT_BOT_META[sid] = meta
        print("METAAAAAAAAAAAAA existing $$$$$$$$$$ ", meta, "$$$$$$$$$$")
        return existing

    # Dataset changed (or no bot yet): drop old bot
    if existing is not None:
        try:
            if hasattr(existing, "cleanup"):
                existing.cleanup()
        except Exception:
            pass
        _SCCHAT_BOTS.pop(sid, None)
        _SCCHAT_BOT_META.pop(sid, None)

    # Create a fresh bot within the session workspace
    session_dir = _session_dir(sid)
    session_dir.mkdir(parents=True, exist_ok=True)

    with _pushd(session_dir):
        # Ensure imports resolve to the real scchatbot package even if session_dir/scchatbot exists
        scchat_root_str = str(SCCHAT_ROOT)
        if scchat_root_str not in sys.path:
            sys.path.insert(0, scchat_root_str)

        try:
            from scchatbot.chatbot import MultiAgentChatBot  # your patched bot
            bot = MultiAgentChatBot()
        except Exception:
            from scchatbot.chatbot import ChatBot  # upstream
            bot = ChatBot()

    _SCCHAT_BOTS[sid] = bot
    _SCCHAT_BOT_META[sid] = {
        "dataset_path": str(dataset_p),
        "dataset_fingerprint": fp,
        "session_dir": str(session_dir),
        "last_access": time.time(),
    }

    print("METAAAAAAAAAAAAA new bot $$$$$$$$$$ ", _SCCHAT_BOT_META[sid], "$$$$$$$$$$")
    return bot




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
    _safe_json_dump(out_path, mapping) # Comment it when you don't want to save the mapping file, else have it for creating mapping file.
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

    media_dir = _session_media_dir(sid)
    media_dir.mkdir(parents=True, exist_ok=True)

    # delete any existing h5ad for this session only
    for old in glob.glob(str(media_dir / "*.h5ad")):
        try:
            os.remove(old)
            print("Deleted old:", old)
        except Exception as e:
            print("Could not delete:", old, e)

    target_path = media_dir / f"{sid}__{safe_name}"
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
    html: Optional[str] = None
    url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    graph: Optional[Dict[str, Any]] = None
    routed_to: Literal["kg", "hybrid"]
    session_id: str
    inquiry_id: Optional[str] = None
    plots: Optional[List[PlotItem]] = None


class AttachRequest(BaseModel):
    session_id: str
    stored_path: str  # path returned by /api/upload (or your own path)

@app.get("/api/scchat_plot/{session_id}/{inquiry_id}/{plot_file}", response_class=HTMLResponse)
async def get_scchat_plot(session_id: str, inquiry_id: str, plot_file: str):
    sid = _sanitize_path_segment(session_id, "session")
    inq = _sanitize_path_segment(inquiry_id, "inquiry")
    fname = Path(plot_file).name
    if not fname.lower().endswith(".html"):
        raise HTTPException(status_code=400, detail="plot_file must be a .html file")

    plots_dir = (_session_dir(sid) / "inquiries" / inq / "plots").resolve()
    target = (plots_dir / fname).resolve()
    if not str(target).startswith(str(plots_dir) + os.sep) and target != plots_dir:
        raise HTTPException(status_code=400, detail="Invalid plot path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(str(target), media_type="text/html")


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

    # Rewrite and reframe the prompt to include metabolic context. Also make the prompt provide the seed entities.
    rewritten_result = await prompt_rewrite(query)
    rewritten_query = rewritten_result.rewritten_prompt
    logger.info("Rewritten question: %s", rewritten_query)

    # Proceed with Hypergraph-BioRAG
    sid = payload.session_id or _new_sid()
    os.chdir(str(APP_CWD))
    # # Reduce retrieved items and context sizes to avoid token overflow
    param = QueryParam(
        top_k=15,
        max_token_for_text_unit=13000,
        max_token_for_global_context=2000,
        max_token_for_local_context=2000,
    )
    base = await rag.aquery(rewritten_query, param)
    base_answer = clean_answer(base)
    
    print(" ######### PROCEEDING TO METABOLIC NETWORK INTEGRATION ######### ")

    # 3) Seeds from query entities_block (exact/near-exact matching happens inside MET runtime)
    seed_text = (rewritten_result.entities_block or "").strip()
    met_ctx, met_json = MET.get_context(seed_text, top_reactions=20)

    if not met_json.get("seeds"):
        fallback_prompt = await prompt_rewrite(base_answer)
        fallback_seed_text = (fallback_prompt.entities_block or "").strip()

        # if the model failed to produce entities_block, fallback to rewritten_prompt (still safer than raw answer)
        if not fallback_seed_text:
            fallback_seed_text = fallback_prompt.rewritten_prompt

        met_ctx, met_json = MET.get_context(fallback_seed_text, top_reactions=20)


    # Final grounded, constraint drive answer rewrite using metabolic network context
    rewrite_result = await rewrite_answer_with_metabolic_grounding(
        question=rewritten_query,
        base_answer=base_answer,
        met_ctx=met_ctx,
        met_json=met_json,
        model="gpt-5.1",
        temperature=0.1,
    )
    print('after metabolic network, new rewrite_result:', rewrite_result)
    
    answer_text = rewrite_result.final_answer

    if isinstance(answer_text, str):
        answer_text = answer_text
    elif isinstance(answer_text, dict):
        answer_text = answer_text.get("answer", str(answer_text))
    else:
        answer_text = str(answer_text)

    final_answer = clean_answer(answer_text)


    # Validator agent:
    # import json, re

    # def extract_claims_json(text: str) -> dict:
    #     m = re.search(r"METABOLIC_CLAIMS_JSON\s*:\s*(\{.*\})", text, flags=re.S)
    #     if not m:
    #         return {}
    #     try:
    #         return json.loads(m.group(1))
    #     except Exception:
    #         return {}

    # def validate_claims(met_json: dict, claims: dict) -> list[str]:
    #     errors = []
    #     rid_map = {r["rid"]: r for r in met_json.get("reactions", [])}

    #     for item in claims.get("used_reactions", []):
    #         rid = item.get("rid")
    #         if rid not in rid_map:
    #             errors.append(f"Reaction {rid} not in curated retrieved set.")
    #             continue
    #         r = rid_map[rid]
    #         if r.get("irreversible") and item.get("direction") == "reverse":
    #             errors.append(f"{rid} is irreversible but used in reverse.")
    #         # compartment mismatch check
    #         if item.get("compartment") and r.get("compartment") and item["compartment"] != r["compartment"]:
    #             errors.append(f"{rid} compartment mismatch: claim={item['compartment']} curated={r['compartment']}")
    #     return errors

    # if isinstance(result, str):
    #     answer_text = result
    # elif isinstance(result, dict):
    #     answer_text = result.get("answer", str(result))
    # else:
    #     answer_text = str(result)

    # answer_text = clean_answer(answer_text)

    graph_json = await build_subgraph_for_query(
        rag=rag,
        query_text=rewritten_query,
        top_k=10,
        hops=2,
        max_entities=50,
        max_hyperedges=50,
    )

    return QueryResponse(
        answer=final_answer,
        graph=graph_json,
        routed_to="kg",
        session_id=sid,
        plots=[],
    )






# ----------------------------
# Hybrid endpoint: upload + scChat + HyperGraphRAG
# scChat has dynamic files - which are changed and added/removed at every use. So when a new bot is initialized, have a copy of the dynamic files created for that session seprate than the static codebase fiLES.
# Single-worker serialized by PIPELINE_LOCK for one user at a time processing.
# ----------------------------
@app.post("/api/query_with_file", response_model=QueryResponse)
async def query_with_file(
    file: UploadFile = File(...),
    query: str = Form(...),
    session_id: Optional[str] = Form(None),
    inquiry_id: Optional[str] = Form(None),
):
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    sid = session_id or _new_sid()
    server_inquiry_id = _sanitize_path_segment(inquiry_id, _sanitize_path_segment(f"{int(time.time())}", "inquiry"))

    # Save/overwrite the session file
    file_path = _save_upload_for_session(sid, file)
    _require_h5ad(file_path)

    # Auto-create mapping for this uploaded h5ad
    media_dir = _session_media_dir(sid)
    mapping_path = await asyncio.to_thread(
        _auto_create_sample_mapping_json,
        sid,
        file_path,
        media_dir,
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
        _cleanup_expired_scchat_sessions()

        session_dir = _session_dir(sid)
        session_dir.mkdir(parents=True, exist_ok=True)

        # scChat uses many relative paths ("media/...", "conversation_history/...", etc.)
        # so we pin cwd to a per-session workspace to avoid cross-session overlap.
        with _pushd(session_dir):
            os.environ["SCCHAT_RUN_STARTUP_ANNOTATION"] = "0"
            bot = _get_or_create_scchat_bot(sid, file_path)
            print('\n\n Getting bot and starting pipeline for session', sid)

            fp = _file_fingerprint(Path(file_path))
            meta = _SCCHAT_BOT_META.get(sid) or {}

            # (Re-)attach only when the dataset content changes for this session
            if meta.get("attached_fingerprint") != fp:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(bot.attach_dataset, sid, file_path),
                        timeout=SCCHAT_TIMEOUT_SEC,
                    )
                    meta["attached_fingerprint"] = fp
                    meta["last_access"] = time.time()
                    _SCCHAT_BOT_META[sid] = meta
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

        # Persist plots per-inquiry (no overwriting between inquiries)
        # Keep this separate from scChat's session workspace so future runs can clear scchatbot/* without losing history.
        inquiry_dir = _ensure_unique_dir(_inquiry_dir(sid, server_inquiry_id))
        server_inquiry_id = inquiry_dir.name  # in case we had to suffix for uniqueness
        plots_dir = (inquiry_dir / "plots").resolve()
        plots_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "session_id": sid,
            "inquiry_id": server_inquiry_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "num_plots": len(scchat_plots or []),
        }

        plot_items: List[Dict[str, str]] = []
        for idx, p in enumerate(scchat_plots or []):
            title = str(getattr(p, "title", "") or "Plot")
            slug = _sanitize_path_segment(title.lower(), f"plot-{idx+1}")
            fname = f"plot_{idx+1:02d}_{slug}.html"
            out_path = (plots_dir / fname).resolve()
            out_path.write_text(str(getattr(p, "html", "") or ""), encoding="utf-8")
            plot_items.append({"title": title, "file": fname})

            p.url = f"/api/scchat_plot/{sid}/{server_inquiry_id}/{fname}"
            p.html = None  # avoid huge response bodies; frontend fetches by URL

        meta["plots"] = plot_items
        try:
            (inquiry_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

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

        # Persist inquiry outputs (answer + graph) alongside plots
        try:
            (inquiry_dir / "query.txt").write_text(q, encoding="utf-8")
        except Exception:
            pass
        try:
            (inquiry_dir / "answer.txt").write_text(answer_text, encoding="utf-8")
        except Exception:
            pass
        try:
            (inquiry_dir / "graph.json").write_text(json.dumps(graph_json, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        try:
            meta_path = inquiry_dir / "meta.json"
            if meta_path.exists():
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                existing_meta = {}
            existing_meta.update(
                {
                    "routed_to": "hybrid",
                    "file_path": str(file_path),
                    "file_name": Path(file_path).name,
                    "latency_sec": None,
                }
            )
            meta_path.write_text(json.dumps(existing_meta, indent=2), encoding="utf-8")
        except Exception:
            pass

        return QueryResponse(
            answer=answer_text,
            graph=graph_json,
            routed_to="hybrid",
            session_id=sid,
            inquiry_id=server_inquiry_id,
            plots=scchat_plots,
        )

    finally:
        PIPELINE_LOCK.release()
