# AGENTS.md

The seat pack (`USER.md`, `MEMORY.md`, atoms) is not this tree.
Do not embed git-tracked files into memory.

## Commands

- `cargo test --workspace --exclude packset-milli` — the writer
- `just surface` — the same requests against both writers, answers compared
- `just interop` — both writers over one store, in both directions
- `just goldens` — regenerate the corpus the Rust port is checked against
- `just test-py` — pytest on `scripts/test_*.py`
- `just lint` — ruff
- `packset ensure` / `packset status`

The search binary is built on the remote builder. `just milli`
refuses anywhere else. Search falls back to the linear scorer
when the binary is absent.

## Architecture

- `crates/packset-daemon` — the writer; atoms in LMDB, cards on disk
- `scripts/packsetd.py` — the reference the writer is checked against,
  not the writer. `packset which` says which one a seat starts.
- `crates/packset-core` — schema, prose, recall, scoring, Borda, MMR,
  decay, extract filters
- `crates/packset-client` — HTTP
- `crates/packset-milli` — search projection

Listen on `127.0.0.1` only. Never `localhost`.

Atoms are one claim. `Remember:` / `Prefer:` are instant.
Tool dumps are attach, not atoms.
