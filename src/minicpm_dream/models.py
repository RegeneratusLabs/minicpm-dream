"""MiniCPM model registry, download and checksum verification."""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Config
from .util.atomic import ensure_dir
from .util.logging import get_logger

log = get_logger("models")

HF = "https://huggingface.co"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    filename: str
    size: int
    sha256: str
    context: int
    license: str
    notes: str = ""

    @property
    def url(self) -> str:
        return f"{HF}/{self.repo}/resolve/main/{self.filename}?download=true"


MODELS: dict[str, ModelSpec] = {
    "minicpm5-2b-q4": ModelSpec(
        key="minicpm5-2b-q4",
        repo="openbmb/MiniCPM5-2B-GGUF",
        filename="MiniCPM5-2B-Q4_K_M.gguf",
        size=1_561_318_368,
        sha256="ec2d5801640099e97d8d7e8003ad4d81f336e757811f03a26173dddf386602fd",
        context=131_072,
        license="Apache-2.0",
        notes="Default: 2.52B params, 131K context, standard LlamaForCausalLM.",
    ),
    "minicpm5-2b-q8": ModelSpec(
        key="minicpm5-2b-q8",
        repo="openbmb/MiniCPM5-2B-GGUF",
        filename="MiniCPM5-2B-Q8_0.gguf",
        size=2_679_710_688,
        sha256="",
        context=131_072,
        license="Apache-2.0",
        notes="Higher-fidelity 2B; ~2.7 GB.",
    ),
    "minicpm5-1b-q4": ModelSpec(
        key="minicpm5-1b-q4",
        repo="openbmb/MiniCPM5-1B-GGUF",
        filename="MiniCPM5-1B-Q4_K_M.gguf",
        size=688_065_920,
        sha256="",
        context=131_072,
        license="Apache-2.0",
        notes="Cheap triage / verifier tier.",
    ),
    "minicpm4.1-8b-q4": ModelSpec(
        key="minicpm4.1-8b-q4",
        repo="openbmb/MiniCPM4.1-8B-GGUF",
        filename="MiniCPM4.1-8B-Q4_K_M.gguf",
        size=4_965_526_048,
        sha256="",
        context=131_072,
        license="Apache-2.0",
        notes="Quality tier; use a GPU or a long weekly pass.",
    ),
}


def get_spec(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model '{key}'; choose from {', '.join(sorted(MODELS))}")
    return MODELS[key]


def resolve_model_path(cfg: Config) -> Path:
    if cfg.model.model_path:
        return Path(cfg.model.model_path)
    spec = get_spec(cfg.model.name)
    return cfg.models_dir / spec.filename


def human_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num} B"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    spec: ModelSpec,
    dest: str | Path,
    *,
    verify: bool = True,
    token: str | None = None,
    force: bool = False,
) -> Path:
    dest = Path(dest).expanduser()
    ensure_dir(dest.parent)
    if dest.exists() and not force:
        if dest.stat().st_size == spec.size:
            log.info("model already present: %s", dest)
            return dest
        log.warning("model exists with unexpected size (%s); re-downloading", human_size(dest.stat().st_size))
    temp = dest.with_suffix(dest.suffix + ".part")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    timeout = httpx.Timeout(60.0, read=None)
    log.info("downloading %s (%s)", spec.filename, human_size(spec.size))
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
        with client.stream("GET", spec.url) as response:
            if response.status_code >= 400:
                raise RuntimeError(
                    f"download failed ({response.status_code}) for {spec.url}; "
                    "check network/HF access or use --model-path"
                )
            total = int(response.headers.get("content-length") or spec.size)
            done = 0
            digest = hashlib.sha256()
            with temp.open("wb") as fh:
                for chunk in response.iter_bytes(1024 * 1024):
                    fh.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if sys.stderr.isatty():
                        pct = done * 100 / max(total, 1)
                        sys.stderr.write(f"\r  {pct:5.1f}%  {human_size(done)} / {human_size(total)}")
                        sys.stderr.flush()
    if sys.stderr.isatty():
        sys.stderr.write("\n")
    actual_size = temp.stat().st_size
    if actual_size != spec.size:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded size {actual_size} != expected {spec.size}")
    if verify:
        actual = digest.hexdigest()
        expected = spec.sha256 or os.environ.get("MINICPM_DREAM_MODEL_SHA256", "")
        if expected and actual != expected:
            temp.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {spec.filename}: got {actual}, expected {expected}. "
                "The file was deleted. Set MINICPM_DREAM_MODEL_SHA256 to the published hash if "
                "the upstream file was legitimately re-released."
            )
        if not expected:
            log.warning("no pinned SHA-256 for %s; size verified only", spec.filename)
    temp.replace(dest)
    os.chmod(dest, 0o600)
    log.info("model ready: %s (%s)", dest, human_size(actual_size))
    return dest


def model_status(cfg: Config) -> dict[str, object]:
    try:
        spec = get_spec(cfg.model.name)
    except KeyError as exc:
        return {"ok": False, "error": str(exc)}
    path = resolve_model_path(cfg)
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    return {
        "ok": exists and size == spec.size,
        "key": spec.key,
        "path": str(path),
        "exists": exists,
        "size": size,
        "expected_size": spec.size,
        "sha256": spec.sha256,
        "context": spec.context,
        "license": spec.license,
    }
