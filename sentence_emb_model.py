import os
import sys
import math
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch

#
# Work around ModernBERT's use of torch.compile on Python 3.12 with torch 2.3
# where Dynamo is not supported and raises at import time.
# We replace torch.compile with a no-op decorator/identity in that environment
# before importing sentence_transformers/transformers model code.
#
if sys.version_info >= (3, 12):
    try:
        if hasattr(torch, "compile"):
            def _fake_torch_compile(*args, **kwargs):
                # Usage 1: compiled_fn = torch.compile(fn)
                if args and callable(args[0]) and len(args) == 1 and not kwargs:
                    return args[0]
                # Usage 2: @torch.compile(dynamic=True) -> returns a decorator
                def _decorator(fn):
                    return fn
                return _decorator
            torch.compile = _fake_torch_compile  # type: ignore[attr-defined]
    except Exception:
        pass

try:
    from sentence_transformers import SentenceTransformer, models
except Exception as e:
    raise RuntimeError(
        "sentence-transformers is required. Please ensure it is installed "
        "(e.g., added to requirements.txt and synced)."
    ) from e


# Use the project-wide logger
logger = logging.getLogger("hypergraphrag")

# Prefer parallel tokenization for speed on CPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# Globals guarded by lazy-init
_MODEL: Optional[SentenceTransformer] = None
_MODEL_DIR="/home/ubuntu/data/pretrained_models/lokeshch19_ModernPubMedBERT"
_POOL = None


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _configure_cpu_threads() -> None:
    """Configure CPU thread counts for best throughput."""
    # Allow override via env; otherwise use all available CPU threads
    num_threads = int(os.environ.get("EMB_NUM_THREADS", str(os.cpu_count() or 1)))
    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass
    # Also hint to BLAS libs if present
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))


def _resolve_model_dir(model_dir: Optional[str] = None) -> str:
    """
    Resolve the local checkpoint directory for modernpubmedbert.
    Priority:
      1) Explicit argument
      2) ENV MODERNPUBMEDBERT_DIR
      3) Common local fallbacks under the repo
    """
    if model_dir:
        p = Path(model_dir).expanduser().resolve()
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Model directory not found: {p}")

    env_path = os.environ.get("MODERNPUBMEDBERT_DIR")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Model directory not found (env): {p}")

    # Fallbacks (do not fail if missing; we will raise after trying)
    repo_root = Path(__file__).parent
    candidates = [
        repo_root / "checkpoints" / "modernpubmedbert",
        repo_root / "models" / "modernpubmedbert",
        Path("/home/ubuntu/models/modernpubmedbert"),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    raise FileNotFoundError(
        "Could not locate modernpubmedbert local checkpoint directory. "
        "Set MODERNPUBMEDBERT_DIR or pass model_dir to get_model()."
    )


def _load_sentence_transformer(model_dir: str, max_seq_length: int) -> SentenceTransformer:
    """
    Load a SentenceTransformer from a local directory.
    If the directory is a plain HF transformer checkpoint, build a Pooling head.
    """
    try:
        model = SentenceTransformer(model_dir, device=_get_device())
        # Some ST checkpoints support setting max length on the word embedding module
        try:
            model.max_seq_length = max_seq_length
        except Exception:
            pass
        return model
    except Exception:
        # Fallback: construct from a plain transformer + pooling
        word = models.Transformer(str(model_dir), max_seq_length=max_seq_length)
        pooling = models.Pooling(
            word.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        model = SentenceTransformer(modules=[word, pooling], device=_get_device())
        return model


def get_model(model_dir: Optional[str] = None, max_seq_length: int = 256) -> SentenceTransformer:
    """
    Lazy-load and cache the sentence embedding model.
    """
    global _MODEL, _MODEL_DIR
    if _MODEL is not None:
        return _MODEL

    if model_dir is None:
        model_dir = _MODEL_DIR

    _configure_cpu_threads()
    logger.info(f"Loading modernpubmedbert from: {model_dir}")
    model = _load_sentence_transformer(model_dir, max_seq_length=max_seq_length)
    model.eval()
    # Ensure we stay on CPU if no GPU
    if _get_device() == "cpu":
        model.to("cpu")
    _MODEL = model
    _MODEL_DIR = model_dir
    return _MODEL


def embed_one(text: str, normalize: bool = True, max_seq_length: int = 256) -> np.ndarray:
    """
    Encode a single sentence into an embedding vector.
    """
    model = get_model(max_seq_length=max_seq_length)
    emb = model.encode(
        [text],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        device=_get_device(),
    )
    return emb[0]


def _batched(iterable: Iterable[str], batch_size: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_batch(
    texts: Sequence[str],
    batch_size: int = 128,
    normalize: bool = True,
    max_seq_length: int = 256,
) -> np.ndarray:
    """
    Encode a list of sentences and return a single numpy array of shape (N, D).
    Suitable when N fits in memory.
    """
    model = get_model(max_seq_length=max_seq_length)
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        device=_get_device(),
    )


def embed_iter(
    sentences: Iterable[str],
    batch_size: int = 128,
    normalize: bool = True,
    max_seq_length: int = 256,
) -> Iterator[np.ndarray]:
    """
    Streaming encoder: yields embeddings for each batch as numpy arrays.
    Use this for very large corpora (millions/billions).
    """
    model = get_model(max_seq_length=max_seq_length)
    for batch in _batched(sentences, batch_size):
        yield model.encode(
            batch,
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            device=_get_device(),
        )


def start_multiprocessing_pool(num_processes: Optional[int] = None) -> None:
    """
    Start a SentenceTransformers multi-process pool for faster CPU throughput.
    Use with embed_iter_multiprocess. Stop with stop_multiprocessing_pool().
    """
    global _POOL
    if _POOL is not None:
        return
    model = get_model()
    if num_processes is None:
        # Reasonable default; can tune via env EMB_NUM_PROCS
        num_processes = int(os.environ.get("EMB_NUM_PROCS", str(max(1, (os.cpu_count() or 1) // 2))))
    _POOL = model.start_multi_process_pool(target_devices=["cpu"] * num_processes)
    logger.info(f"Started embedding pool with {num_processes} processes")


def stop_multiprocessing_pool() -> None:
    global _POOL
    if _POOL is not None:
        model = get_model()
        model.stop_multi_process_pool(_POOL)
        _POOL = None
        logger.info("Stopped embedding pool")


def embed_iter_multiprocess(
    sentences: Iterable[str],
    batch_size: int = 4096,
    normalize: bool = True,
    max_seq_length: int = 256,
) -> Iterator[np.ndarray]:
    """
    Streaming multiprocess encoder. Requires start_multiprocessing_pool() first.
    Feeds manageable chunks into encode_multi_process and yields batch embeddings.
    """
    if _POOL is None:
        raise RuntimeError("Multiprocessing pool not started. Call start_multiprocessing_pool() first.")
    model = get_model(max_seq_length=max_seq_length)
    for batch in _batched(sentences, batch_size):
        arr = model.encode_multi_process(
            sentences=list(batch),
            pool=_POOL,
            batch_size=max(64, min(1024, len(batch))),
            normalize_embeddings=normalize,
        )
        yield arr


if __name__ == "__main__":
    # Minimal CLI for quick ad-hoc testing:
    import argparse

    parser = argparse.ArgumentParser(description="Encode sentences with modernpubmedbert (local checkpoint).")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to local modernpubmedbert checkpoint.")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for encoding.")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings.")
    parser.add_argument("--text", type=str, default=None, help="Single sentence to encode.")
    parser.add_argument("--file", type=str, default=None, help="Path to a text file with one sentence per line.")
    args = parser.parse_args()

    if args.model_dir:
        os.environ["MODERNPUBMEDBERT_DIR"] = args.model_dir

    if args.text:
        vec = embed_one(args.text, normalize=args.normalize, max_seq_length=args.max_seq_length)
        np.set_printoptions(precision=6, suppress=True)
        print(vec)
        sys.exit(0)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = (line.rstrip("\n") for line in f)
            total = 0
            for out in embed_iter(lines, batch_size=args.batch_size, normalize=args.normalize, max_seq_length=args.max_seq_length):
                total += out.shape[0]
                # Write or aggregate downstream; here we just count
            print(f"Encoded {total} sentences")
        sys.exit(0)

    parser.print_help()

