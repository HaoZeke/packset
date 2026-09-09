# AGENTS.md

The seat pack (`USER.md`, `MEMORY.md`, atoms) is not this tree.
Do not embed git-tracked files into memory.

## Commands

- `just check` — fmt, clippy, and the workspace tests
- `just bench` — what a pack costs as it grows, and its link degree
- `cargo run --release -p packset-daemon --example locomo -- locomo10.json`
  — retrieval quality against LoCoMo's labelled evidence
- `packset ensure` / `packset status`

The search binary is built on the remote builder. `just milli`
refuses anywhere else. Search falls back to the linear scorer
when the binary is absent.

## Architecture

- `crates/packset-daemon` — the writer; atoms in LMDB, cards on disk
- `crates/packset-cli` — `packset`: lifecycle and the `/v1` reads a
  seat runs from a shell. `bin/packset` execs whichever build exists.
- `crates/packset-core` — schema, prose, recall, scoring, BM25, Borda,
  MMR, decay, extract filters
- `crates/packset-client` — HTTP
- `crates/packset-milli` — search projection
- `crates/packset-embed` — dense projection, a kept child process so the
  model loads once rather than per question

Listen on `127.0.0.1` only. Never `localhost`.

Atoms are one claim. `Remember:` / `Prefer:` are instant.
Tool dumps are attach, not atoms.

`crates/packset-core/tests/goldens.json` is a frozen corpus, not
generated output. It fixes the accept/reject boundary and the exact error
strings, so a change to it is a change in what the daemon accepts.
