# full-day scenario

24 turns across three days, 14 gold facts. It exercises:

- identity, preferences, projects, people, environment, procedures, lessons;
- an update (SQLite → PostgreSQL) that must supersede rather than duplicate;
- explicit dates (2026-11-03, 2026-11-04, 2026-11-06) for temporal scoring;
- a retraction ("no longer use Ollama") that must not become a fact;
- a Chinese turn that must be stored in English;
- a meta-state, an injection, and an API key that must be rejected/dropped.

This is a regression suite, not a benchmark. The mock backend covers only
`basic/`; run `full-day/` with a real model:

    minicpm-dream eval --suite evals/synthetic-days/full-day --backend llama_cpp
