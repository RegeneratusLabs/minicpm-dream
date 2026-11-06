"""Deterministic episode filters applied before any model call."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import IngestCfg, PrivacyCfg
from ..util.text import normalize_text
from .base import Episode
from .redact import injection_flags, redact


@dataclass
class FilterStats:
    seen: int = 0
    kept: int = 0
    dropped_source: int = 0
    dropped_role: int = 0
    dropped_empty: int = 0
    dropped_duplicate: int = 0
    dropped_long: int = 0
    dropped_injection: int = 0
    dropped_secret: int = 0
    redactions: int = 0
    flags: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int]:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "dropped_source": self.dropped_source,
            "dropped_role": self.dropped_role,
            "dropped_empty": self.dropped_empty,
            "dropped_duplicate": self.dropped_duplicate,
            "dropped_long": self.dropped_long,
            "dropped_injection": self.dropped_injection,
            "dropped_secret": self.dropped_secret,
            "redactions": self.redactions,
        }


class EpisodeFilter:
    def __init__(self, cfg: IngestCfg, privacy: PrivacyCfg | None = None):
        self.cfg = cfg
        self.privacy = privacy or PrivacyCfg()
        self.stats = FilterStats()
        self._hashes: set[str] = set()

    def keep(self, ep: Episode) -> bool:
        self.stats.seen += 1
        if ep.source in self.cfg.exclude_sources:
            self.stats.dropped_source += 1
            return False
        if ep.source not in self.cfg.trusted_sources:
            self.stats.dropped_source += 1
            return False
        allowed_roles = {"user", "assistant"}
        if self.cfg.include_tool_messages:
            allowed_roles.add("tool")
        if ep.role not in allowed_roles:
            self.stats.dropped_role += 1
            return False
        text = normalize_text(ep.text)
        if len(text) < 3:
            self.stats.dropped_empty += 1
            return False
        if len(text) > self.cfg.max_message_chars:
            self.stats.dropped_long += 1
            return False
        if self.privacy.redact:
            result = redact(text)
            self.stats.redactions += sum(result.hits.values())
            ep.text = result.text
            ep.flags.extend(f"redacted:{k}" for k in result.hits)
            sensitive = set(result.hits) - {"email", "phone", "phone_intl"}
            if sensitive and self.privacy.drop_redacted:
                self.stats.dropped_secret += 1
                return False
        if self.privacy.screen_injection:
            flags = injection_flags(ep.text)
            if flags:
                ep.flags.extend(f"injection:{f}" for f in flags)
                self.stats.dropped_injection += 1
                return False
        if self.cfg.dedupe:
            digest = ep.text_hash
            if digest in self._hashes:
                self.stats.dropped_duplicate += 1
                return False
            self._hashes.add(digest)
        self.stats.kept += 1
        for flag in ep.flags:
            self.stats.flags[flag] = self.stats.flags.get(flag, 0) + 1
        return True
