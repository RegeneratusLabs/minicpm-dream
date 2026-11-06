"""OpenBMB retrieval stack: MiniCPM-Embedding-Light + MiniCPM-Reranker-Light.

Optional dependencies: torch, transformers, safetensors, numpy. The models run
locally on CPU or GPU; nothing is sent to a network after the initial download.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..util.logging import get_logger

log = get_logger("retrieval.openbmb")

DEFAULT_EMBEDDING_MODEL = "openbmb/MiniCPM-Embedding-Light"
DEFAULT_RERANKER_MODEL = "openbmb/MiniCPM-Reranker-Light"
DEFAULT_QUERY_PREFIX = "Query: "

_INSTALL_HINT = (
    "OpenBMB retrieval needs the optional stack: "
    "`uv pip install -e '.[openbmb]'` (torch CPU + transformers + safetensors)."
)


class OpenBMBUnavailable(RuntimeError):
    pass


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise OpenBMBUnavailable(_INSTALL_HINT) from exc
    return torch


def _dtype(name: str) -> Any:
    torch = _torch()
    value = name.lower()
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def _from_pretrained(model_cls: Any, model_name: str, dtype: str) -> Any:
    """Support both transformers' `dtype` (new) and `torch_dtype` (old)."""
    try:
        return model_cls.from_pretrained(
            model_name, trust_remote_code=True, dtype=_dtype(dtype)
        )
    except TypeError:
        return model_cls.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=_dtype(dtype)
        )


def _to_numpy(tensor: Any) -> Any:
    import numpy as np

    return tensor.detach().to("cpu").float().numpy().astype(np.float32)


@dataclass
class EmbeddingBatch:
    dense: Any  # numpy array [n, dim]
    sparse: list[dict[int, float]] | None = None


class OpenBMBEmbedder:
    """Dense (and optional sparse) embeddings from MiniCPM-Embedding-Light."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        max_length: int = 512,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        threads: int = 4,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype_name = dtype
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.threads = threads
        self._model: Any = None
        self._tokenizer: Any = None
        self.dimension: int | None = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def load(self) -> OpenBMBEmbedder:
        if self._model is not None:
            return self
        torch = _torch()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise OpenBMBUnavailable(_INSTALL_HINT) from exc
        try:
            torch.set_num_threads(max(1, self.threads))
        except Exception:  # noqa: BLE001
            pass
        log.info("loading embedding model %s on %s (%s)", self.model_name, self.device, self.dtype_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = _from_pretrained(AutoModel, self.model_name, self.dtype_name).to(self.device)
        self._model.eval()
        hidden = getattr(self._model.config, "hidden_size", None)
        self.dimension = int(hidden) if hidden else None
        return self

    def _encode(self, texts: Sequence[str]) -> Any:
        if not texts:
            import numpy as np

            return np.zeros((0, self.dimension or 0), dtype=np.float32)
        self.load()
        torch = _torch()
        import numpy as np

        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            output = self._model(**encoded)
        hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return _to_numpy(pooled)

    def encode_queries(self, texts: Sequence[str]) -> Any:
        return self._encode([f"{self.query_prefix}{text}" for text in texts])

    def encode_corpus(self, texts: Sequence[str]) -> Any:
        return self._encode(list(texts))

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            _torch().cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


class OpenBMBReranker:
    """Cross-encoder reranker from MiniCPM-Reranker-Light."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        max_length: int = 512,
        threads: int = 4,
        batch_size: int = 4,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype_name = dtype
        self.max_length = max_length
        self.threads = threads
        self.batch_size = batch_size
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> OpenBMBReranker:
        if self._model is not None:
            return self
        torch = _torch()
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise OpenBMBUnavailable(_INSTALL_HINT) from exc
        try:
            torch.set_num_threads(max(1, self.threads))
        except Exception:  # noqa: BLE001
            pass
        log.info("loading reranker %s on %s (%s)", self.model_name, self.device, self.dtype_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = _from_pretrained(
            AutoModelForSequenceClassification, self.model_name, self.dtype_name
        ).to(self.device)
        self._model.eval()
        return self

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        self.load()
        torch = _torch()
        pairs = [[f"Query: {query}", doc] for doc in documents]
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self._tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                output = self._model(**encoded)
            logits = output.logits
            if logits.dim() == 2:
                logits = logits[:, 0]
            scores.extend(float(value) for value in logits.detach().to("cpu").float().tolist())
        return scores

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            _torch().cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def download_models(
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
) -> dict[str, str]:
    """Prefetch weights into the Hugging Face cache; no model is loaded."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise OpenBMBUnavailable(_INSTALL_HINT) from exc
    out: dict[str, str] = {}
    for repo in (embedding_model, reranker_model):
        log.info("downloading %s", repo)
        out[repo] = snapshot_download(repo_id=repo)
    return out


def cache_status(model_name: str) -> str:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return "unknown"
    path = try_to_load_from_cache(model_name, "config.json")
    return str(path) if isinstance(path, str) else "not cached"


def build_embedder(config: Any) -> OpenBMBEmbedder:
    return OpenBMBEmbedder(
        config.embedding_model,
        device=config.embedding_device,
        dtype=config.embedding_dtype,
        max_length=config.embedding_max_length,
        threads=config.embedding_threads,
    )


def build_reranker(config: Any) -> OpenBMBReranker:
    return OpenBMBReranker(
        config.reranker_model,
        device=config.embedding_device,
        dtype=config.reranker_dtype,
        max_length=config.reranker_max_length,
        threads=config.embedding_threads,
        batch_size=config.reranker_batch_size,
    )
