from __future__ import annotations

from minicpm_dream.synth.dream_md import append_section, quiet_section


def test_appended_sections_do_not_double_the_heading(tmp_path):
    """Regression: render_section() already leads with "## ", and append_section()
    rejoined with the same prefix, so every run after the first wrote "## ## ..."."""
    path = tmp_path / "DREAMS.md"
    append_section(path, "## 2026-11-08 03:00 — day one\n\n### Promoted\n- a\n")
    append_section(path, "## 2026-11-09 03:00 — day two\n\n### Promoted\n- b\n")
    quiet = quiet_section(
        started_at="2026-11-10 03:00",
        window="day three",
        run_id="run_x",
        counters={"episodes_kept": 4, "candidates": 0},
    )
    append_section(path, quiet)

    text = path.read_text(encoding="utf-8")
    assert "## ## " not in text
    assert text.count("\n## ") == 3
    assert text.index("day one") < text.index("day two") < text.index("day three")
