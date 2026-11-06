# Toy harness

A five-message "day" with three durable facts and one meta-state that the
extractor should ignore. The mock script is canned: it proves the pipeline
(ingest → chunk → extract → verify → reconcile → ledger → diary → exports)
without downloading a model.

```bash
make demo
```

or manually:

```bash
minicpm-dream run \
  --source jsonl \
  --input examples/toy-harness/episodes.jsonl \
  --backend mock \
  --mock-script examples/toy-harness/mock_script.json \
  --data-dir ./demo-data \
  --since 2026-11-04T14:00:00+00:00 \
  --until 2026-11-05T14:00:00+00:00

minicpm-dream search "dentist" --data-dir ./demo-data
cat ./demo-data/DREAMS.md
cat ./demo-data/exports/review.json
```

Expected:

- 4 messages kept (the meta-state is dropped by validation, not by ingest);
- 3 candidates, all verified;
- 3 memories: a preference, a project fact, and a dated task on 2026-11-06;
- 2 Hermes review actions (the task is `open_loops`, diary-only);
- no writes outside `./demo-data`.

To run the same fixture with the real model:

```bash
minicpm-dream setup --download
LLAMA_SERVER=/path/to/llama-server minicpm-dream run \
  --source jsonl --input examples/toy-harness/episodes.jsonl \
  --backend llama_cpp --data-dir ./demo-real
```

The real model may phrase things differently than the mock; what matters is that
the same invariants hold (quotes, dates, no invented targets).
