# Privacy and data handling

This tool is designed to run on the user's own machine with a local model. The
default path makes no network calls after installation.

## Data flow

```
Hermes state.db ──(read-only)──► adapter ──► redaction ──► model prompt
                                                     │
                                                     ├──► temp ledger (dry-run)
                                                     └──► SQLite ledger + DREAMS.md + exports
```

- **Read**: `state.db`, `MEMORY.md`, `USER.md`, `config.yaml` (limits only).
- **Sent to the model**: redacted episode text, date refs, and retrieved memory
  texts. Nothing else.
- **Network**: only the explicit `setup --download` / `--download-embeddings`
  (Hugging Face), or a backend you configure (`openai` with a `base_url`). There
  is no telemetry. Embedding and reranking run locally.
- **Written**: the ledger, diary, exports, backups, and logs. No writes to
  Hermes unless `--apply-hermes` and `write_mode = "apply"`.

## What is stored

| Store | Content | Default |
|---|---|---|
| `episodes` | normalized turns; `text_redacted` only | `privacy.store_raw = "none"` → text column null |
| `memories` | memory statements | yes |
| `evidence` | verbatim quotes (redacted) | yes |
| `ops` / `proposals` | candidate payloads, decisions, rationale | yes |
| `DREAMS.md` | promoted/changed items + one evidence quote | yes |
| `exports/` | Hermes-format candidates, `review.json` | yes |
| `backups/` | copies of Hermes memory before a direct write | only on apply |

`privacy.store_raw` accepts:

- `"none"` (default): only memory text and evidence quotes persist, not the full
  turns.
- `"redacted"`: store the full redacted turn for audit and re-runs.
- `"full"`: currently the same as `"redacted"` — the adapter never stores the
  unredacted source. The name is kept for forward compatibility.

## Redaction

Before any model call, these are replaced with `[REDACTED:<kind>]`:

- private keys (PEM blocks), OpenAI/Anthropic/GitHub/Slack/AWS/Google token
  shapes, JWTs;
- `api_key: …`, `token=…`, `password=…` assignments;
- emails;
- phone numbers when labelled (`phone:`) or in international `+` form.

This is a **filter, not a guarantee**. A novel credential shape can pass. The
ledger and exports are `0600` in a `0700` directory, but they are not encrypted.
If the host is shared, use full-disk encryption and a dedicated OS account.

## Prompt injection

Transcripts are data, never instructions:

- obvious injection patterns and invisible Unicode are detected and the episode
  is quarantined before prompting;
- the extractor prompt explicitly says the episodes are data;
- model output is parsed as JSON and validated; it is never executed as a
  command, never used to build a path, and never used to select a tool;
- target IDs are constrained to the retrieval set;
- secrets and meta-states are rejected at write time.

A successful injection could still waste a model call or produce a rejected
candidate. It cannot cause a shell command, file write outside the configured
paths, or a Hermes write without the explicit apply path.

## Retention and forgetting

- Memories and evidence persist until explicitly forgotten or archived.
- `minicpm-dream forget <id> --yes`:
  - sets `status='tombstone'`;
  - replaces text with `[forgotten]`;
  - replaces evidence quotes with `[forgotten]` and clears the vector;
  - keeps the id and a KV marker for audit (no text).
- `maintain --archive-days N` archives low-confidence, long-unused memories
  outside identity/preferences. It does not delete them.
- Backups of Hermes files are **not** rewritten by `forget`; remove them
  separately if you need a hard erase. The same applies to filesystem snapshots,
  database dumps, and shell history.
- `privacy.retention_days` is reserved; v1 does not auto-purge.

## Threat model

| Threat | Mitigation |
|---|---|
| Secret in a chat gets stored | pre-model redaction + post-validation secret rejection |
| Prompt injection in a transcript | quarantine + data-only prompt + no tool execution |
| Model hallucinates a fact | verbatim quote requirement + entailment verifier |
| Model invents a merge detail | digit/date novelty guard; default DEFER |
| Run destroys memory | bitemporal supersede, no hard delete, 25% loss guard, transaction rollback |
| Hermes memory clobbered | opt-in apply, lock, mtime check, backup, char-limit check |
| Concurrent writers | run lock + Hermes memory lock |
| Schema drift misreads data | introspection, explicit warnings, JSONL fallback |
| Model endpoint sees data | local by default; remote backend is an explicit config choice |

## Recommendations

- Keep `hermes.write_mode = "proposals"` until you have reviewed several nights.
- Keep `memory.write_approval: true` in Hermes if other agents share the profile.
- Use a dedicated OS user for the agent and the consolidator.
- Back up the data directory; the ledger is the source of truth, `DREAMS.md` is
  a view.
- Review `DREAMS.md` weekly; it is the audit trail for what the model believed.
- `scripts/compare_models.py` sends the chosen suite to a cloud endpoint. It
  defaults to the synthetic suite; do not point it at private transcripts
  without deciding that transfer is acceptable.
