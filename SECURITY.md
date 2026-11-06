# Security policy

`minicpm-dream` runs locally and treats agent transcripts as untrusted data.
The threat model and mitigations are documented in [docs/privacy.md](docs/privacy.md).

## Reporting

Please report suspected vulnerabilities privately to the maintainers rather than
opening a public issue with exploit details. Do not include real transcripts,
credentials, or personal data in a report; use synthetic reproductions.

## Scope

In scope:

- secret leakage through prompts, logs, the ledger, diary, or exports;
- prompt-injection paths that cause filesystem writes, tool execution, or memory
  corruption;
- unsafe writes to Hermes memory;
- authentication or path-traversal issues in adapters.

Out of scope:

- the security of a remote inference endpoint you configure yourself;
- model weights or llama.cpp vulnerabilities (report upstream);
- physical access to the host or an unencrypted disk.

## Defaults

- Hermes is read-only unless `--apply-hermes` and `write_mode = "apply"`.
- Secrets are redacted before prompting and rejected before writing.
- Injection-like turns are quarantined.
- The ledger is bitemporal and never hard-deletes; `forget` leaves a tombstone.
- A 25% active-memory loss rolls the run back.
- Files are written atomically with `0600` in a `0700` data directory.
- No telemetry and no network calls except model downloads or an explicitly
  configured backend.
