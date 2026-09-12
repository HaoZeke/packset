# Changelog

Versions follow semver at 0.x: a minor bump is a feature, a patch is a fix.

## Unreleased

- The decay slot defaults to `fsrs`: retrievability from the review clock,
  floored at 0.25, cards exempt. `PACKSET_DECAY=off` turns it off. On a
  corpus with no review history the slot changes nothing.

## 0.4.0 (2026-09-12)

- `PacksetClient::with_workspace` pins the workspace ahead of
  `PACKSET_WORKSPACE` and the working directory, so a seat can be one memory
  across every repository it works in.

- The client finds the writer at `http://127.0.0.1:8761` when `PACKSET_URL`
  is unset (`PACKSET_PORT` moves the port), the same default the command
  line and the MCP server use, so a seat needs no variable set.
  `PACKSET_URL=off` is the one way to have no pack.

## 0.3.0 (2026-09-12)

What a user gets:

- A review clock on every claim: written claims are due after one day,
  `grade` moves them, `due` lists what is about to be forgotten. With
  `PACKSET_DECAY=fsrs` the FSRS retrievability curve scales search scores,
  so unreviewed claims fade without vanishing.
- Memory islands: `islands` lists the link graph's clusters, `island CUE`
  returns the memories a task activates by spreading activation, and `fire`
  (or `island --fire`) strengthens the links of claims used together, with a
  forgetting term so weights stay bounded.
- Trust rows: a `trust` atom is one weighted edge of an influence graph, with
  a validity window like any claim, exported with the rest.
- The pack's own command line: `remember`, `prefer`, `search`, `due`,
  `grade`, `islands`, `island`, `fire`, beside `ensure`, `status`, `export`.
- A hit's `score` is the panel's fused weight, one scale across ballots; the
  ballot's own score sits beside it as `ballot_score`.
- BM25+ replaces BM25 as the lexical default; passage windows measured; the
  retrieval table names every arm and what lost.
- LongMemEval_S session retrieval and a many-clients hammer are examples
  with their tables in the README.
- One logical write at a time under many clients; the encoder runs before
  the write lock and is warmed at start; the index is rebuilt from cached
  tokens after a write.
- `packset-embed` caches models under `$XDG_CACHE_HOME/packset/embed`, never
  the working directory.
- A sentence boundary needs whitespace after the full stop, so `0.9.3`,
  `127.0.0.1` and `Cargo.lock` no longer count as sentence ends.
- Client requests time out after thirty seconds, settable with
  `PACKSET_TIMEOUT_MS`; refusals carry the daemon's one-line reason.
- A documentation site at https://leidarljos.github.io/packset/.

## 0.2.0

The Rust writer: LMDB store, HTTP surface, the panel of fusers and
diversifiers, the dense encoder as a child process, export for handovers.
