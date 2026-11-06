"""Configuration loading with TOML files, environment overrides and CLI precedence."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

from platformdirs import user_config_dir, user_data_dir

APP_NAME = "minicpm-dream"
APP_AUTHOR = "miniCPM-memory"

ENV_CONFIG = "MINICPM_DREAM_CONFIG"


def detect_timezone() -> str:
    """Best-effort IANA zone; the user can always override with config/env."""
    if os.environ.get("TZ"):
        return os.environ["TZ"]
    try:
        value = Path("/etc/timezone").read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    try:
        link = os.readlink("/etc/localtime")
        if "zoneinfo/" in link:
            return link.split("zoneinfo/", 1)[1]
    except OSError:
        pass
    return "UTC"


def default_data_dir() -> Path:
    return Path(os.environ.get("MINICPM_DREAM_DATA_DIR") or user_data_dir(APP_NAME, APP_AUTHOR))


def default_config_path() -> Path:
    explicit = os.environ.get(ENV_CONFIG)
    if explicit:
        return Path(explicit).expanduser()
    return Path(user_config_dir(APP_NAME, APP_AUTHOR)) / "config.toml"


@dataclass
class PathsCfg:
    data_dir: str = ""
    db: str = ""
    exports: str = ""
    models: str = ""
    logs: str = ""
    dreams: str = ""
    backups: str = ""


@dataclass
class ModelCfg:
    name: str = "minicpm5-2b-q4"
    backend: str = "llama_cpp"  # llama_cpp | openai | mock
    model_path: str = ""
    llama_server: str = ""
    base_url: str = ""
    api_key_env: str = ""
    context: int = 8192
    threads: int = 4
    gpu_layers: int = 0
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout_s: float = 180.0
    startup_timeout_s: float = 120.0
    verify_sha256: bool = True
    disable_thinking: bool = True
    structured_mode: str = "auto"
    reasoning_effort: str = ""


@dataclass
class IngestCfg:
    sources: list[str] = None  # type: ignore[assignment]
    trusted_sources: list[str] = None  # type: ignore[assignment]
    exclude_sources: list[str] = None  # type: ignore[assignment]
    include_tool_messages: bool = False
    max_message_chars: int = 8000
    max_episode_chars: int = 8000
    dedupe: bool = True
    trust_private_chats: bool = True

    def __post_init__(self) -> None:
        if self.sources is None:
            self.sources = ["hermes", "jsonl"]
        if self.trusted_sources is None:
            self.trusted_sources = ["telegram", "desktop", "tui", "cli", "local", "jsonl"]
        if self.exclude_sources is None:
            self.exclude_sources = ["cron", "subagent", "oneshot", "tool"]


@dataclass
class ExtractCfg:
    max_items: int = 6
    max_chunk_tokens: int = 1500
    overlap_ratio: float = 0.1
    chunk_by_session: bool = True
    split_long_messages: bool = True
    temperature: float = 0.2
    max_tokens: int = 1200
    verify: bool = True
    verify_max_tokens: int = 160
    verify_lenient: bool = True
    min_confidence: float = 0.45
    coverage_pass: bool = False
    coverage_max_items: int = 4
    message_fallback: bool = True
    message_fallback_max_messages: int = 20
    message_fallback_min_chars: int = 10
    message_fallback_uncited_only: bool = False
    message_fallback_include_assistant: bool = False
    message_fallback_max_items: int = 3
    message_fallback_max_tokens: int = 400
    profile_pass: bool = False
    profile_max_items: int = 8


@dataclass
class ConsolidateCfg:
    top_k: int = 6
    max_changes_per_run: int = 6
    max_new_per_run: int = 3
    min_confidence: float = 0.55
    similarity_threshold: float = 0.9
    temperature: float = 0.1
    max_tokens: int = 400
    max_calls: int = 40


@dataclass
class RetrievalCfg:
    fts: bool = True
    embeddings: str = "openbmb"  # off | openbmb
    embedding_model: str = "openbmb/MiniCPM-Embedding-Light"
    embedding_device: str = "cpu"
    embedding_dtype: str = "float32"
    embedding_max_length: int = 512
    embedding_threads: int = 4
    embedding_batch_size: int = 8
    reranker: str = "openbmb"  # off | openbmb
    reranker_model: str = "openbmb/MiniCPM-Reranker-Light"
    reranker_dtype: str = "float16"
    reranker_max_length: int = 512
    reranker_batch_size: int = 2
    rerank_top_n: int = 6
    rrf_k: int = 60
    strict: bool = False


@dataclass
class HermesCfg:
    home: str = "~/.hermes"
    state_db: str = ""
    memory_file: str = "memories/MEMORY.md"
    user_file: str = "memories/USER.md"
    write_mode: str = "proposals"  # proposals | apply
    memory_char_limit: int = 2200
    user_char_limit: int = 1375
    trusted_chat_types: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.trusted_chat_types is None:
            self.trusted_chat_types = ["dm"]


@dataclass
class BudgetsCfg:
    max_minutes: float = 20.0
    max_input_tokens: int = 150_000
    max_output_tokens: int = 40_000
    max_episodes: int = 2000


@dataclass
class PrivacyCfg:
    store_raw: str = "none"  # none | redacted | full
    redact: bool = True
    screen_injection: bool = True
    quarantine_dir: str = "quarantine"
    retention_days: int = 0  # 0 = keep ledger forever
    drop_redacted: bool = True


@dataclass
class SynthCfg:
    diary_path: str = ""
    keep_sections: int = 90
    narrative: bool = False


@dataclass
class EvalCfg:
    suite: str = "evals/synthetic-days"
    judge: bool = False


@dataclass
class Config:
    timezone: str = ""
    paths: PathsCfg = None  # type: ignore[assignment]
    model: ModelCfg = None  # type: ignore[assignment]
    ingest: IngestCfg = None  # type: ignore[assignment]
    extract: ExtractCfg = None  # type: ignore[assignment]
    consolidate: ConsolidateCfg = None  # type: ignore[assignment]
    retrieval: RetrievalCfg = None  # type: ignore[assignment]
    hermes: HermesCfg = None  # type: ignore[assignment]
    budgets: BudgetsCfg = None  # type: ignore[assignment]
    privacy: PrivacyCfg = None  # type: ignore[assignment]
    synth: SynthCfg = None  # type: ignore[assignment]
    eval: EvalCfg = None  # type: ignore[assignment]
    config_path: str = ""

    def __post_init__(self) -> None:
        for name, cls in (
            ("paths", PathsCfg),
            ("model", ModelCfg),
            ("ingest", IngestCfg),
            ("extract", ExtractCfg),
            ("consolidate", ConsolidateCfg),
            ("retrieval", RetrievalCfg),
            ("hermes", HermesCfg),
            ("budgets", BudgetsCfg),
            ("privacy", PrivacyCfg),
            ("synth", SynthCfg),
            ("eval", EvalCfg),
        ):
            if getattr(self, name) is None:
                setattr(self, name, cls())

    # -- resolution ---------------------------------------------------------
    def resolve(self) -> Config:
        if not self.timezone:
            self.timezone = detect_timezone()
        data_dir = Path(self.paths.data_dir).expanduser() if self.paths.data_dir else default_data_dir()
        self.paths.data_dir = str(data_dir)
        self.paths.db = str(Path(self.paths.db).expanduser() if self.paths.db else data_dir / "ledger.db")
        self.paths.exports = str(
            Path(self.paths.exports).expanduser() if self.paths.exports else data_dir / "exports"
        )
        self.paths.models = str(
            Path(self.paths.models).expanduser() if self.paths.models else data_dir / "models"
        )
        self.paths.logs = str(Path(self.paths.logs).expanduser() if self.paths.logs else data_dir / "logs")
        self.paths.backups = str(
            Path(self.paths.backups).expanduser() if self.paths.backups else data_dir / "backups"
        )
        self.synth.diary_path = str(
            Path(self.synth.diary_path).expanduser()
            if self.synth.diary_path
            else data_dir / "DREAMS.md"
        )
        self.hermes.home = str(Path(self.hermes.home).expanduser())
        if not self.hermes.state_db:
            self.hermes.state_db = str(Path(self.hermes.home) / "state.db")
        if self.model.model_path:
            self.model.model_path = str(Path(self.model.model_path).expanduser())
        return self

    @property
    def db_path(self) -> Path:
        return Path(self.paths.db)

    @property
    def exports_dir(self) -> Path:
        return Path(self.paths.exports)

    @property
    def models_dir(self) -> Path:
        return Path(self.paths.models)

    @property
    def lock_path(self) -> Path:
        return Path(self.paths.data_dir) / "run.lock"

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if is_dataclass(value):
                out[f.name] = {ff.name: getattr(value, ff.name) for ff in fields(value)}
            else:
                out[f.name] = value
        return out


def _build(cls: type, data: dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints.get(f.name)
        if isinstance(hint, type) and is_dataclass(hint):
            if not isinstance(value, dict):
                raise ValueError(f"config section [{f.name}] must be a table")
            value = _build(hint, value)
        elif hint is bool and isinstance(value, str):
            value = value.strip().lower() in {"1", "true", "yes", "on"}
        elif hint is int and isinstance(value, str):
            value = int(value)
        elif hint is float and isinstance(value, (str, int)):
            value = float(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None, *, data_dir: str | None = None) -> Config:
    """Load TOML config, then apply environment overrides.

    Precedence (low to high): defaults < config file < environment < CLI (applied by caller).
    """
    cfg_path = Path(path).expanduser() if path else default_config_path()
    data: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("rb") as fh:
            data = tomllib.load(fh)
    config = _build(Config, data)
    config.config_path = str(cfg_path) if cfg_path.exists() else ""

    env = os.environ
    if data_dir or env.get("MINICPM_DREAM_DATA_DIR"):
        config.paths.data_dir = data_dir or env["MINICPM_DREAM_DATA_DIR"]
    if env.get("MINICPM_DREAM_MODEL"):
        config.model.name = env["MINICPM_DREAM_MODEL"]
    if env.get("MINICPM_DREAM_MODEL_PATH"):
        config.model.model_path = env["MINICPM_DREAM_MODEL_PATH"]
    if env.get("MINICPM_DREAM_BACKEND"):
        config.model.backend = env["MINICPM_DREAM_BACKEND"]
    if env.get("MINICPM_DREAM_BASE_URL"):
        config.model.base_url = env["MINICPM_DREAM_BASE_URL"]
    if env.get("MINICPM_DREAM_HERMES_HOME"):
        config.hermes.home = env["MINICPM_DREAM_HERMES_HOME"]
    if env.get("MINICPM_DREAM_THREADS"):
        config.model.threads = int(env["MINICPM_DREAM_THREADS"])
    if env.get("MINICPM_DREAM_TZ"):
        config.timezone = env["MINICPM_DREAM_TZ"]
    return config.resolve()
