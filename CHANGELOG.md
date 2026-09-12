# Changelog

Versions follow semver at 0.x: a minor bump is a feature, a patch is a fix.

## 0.6.0 (2026-09-12)

- The writer keeps two query encoders (`PACKSET_EMBED_QUERY_WORKERS`), so
  agents asking at once are answered side by side instead of one behind
  the other; eight concurrent hooks took 281 ms wall on one encoder and
  243 ms on two once warm. The pool is warmed at start, since the first
  use of a cold second encoder cost 770 ms.
- LongMemEval_S over every answerable question: the fused panel reaches
  0.889 hit@1, 0.949 recall@5, 0.981 recall@10 at session granularity,
  against 0.855, 0.914, 0.952 for the lexical ballot.
- `packset hubs` and `GET /v1/hubs`: the claims the link graph turns on,
  a weighted PageRank over the links.
- Two kinds: `prediction` (a voter's forecast on an issue, for the
  surprisingly popular rule) and `rule` (a pattern with a verdict, argv law
  kept in the pack and exported with the rest).

## 0.5.1 (2026-09-12)

- Two panel tests were red at 0.5.0: `Panel::parse` still filled the decay
  slot with `off`, and a test read the default as `off`.
- The first answer-accuracy row: with Qwen2.5-7B-Instruct Q5_K_M as
  reader and judge and LongMemEval's own prompts, the fused panel answers
  0.630 of the first hundred questions, the lexical ballot 0.570, the
  labelled sessions 0.690.
- A cross-encoder rerank arm on LongMemEval (`PACKSET_LME_RERANK=1`),
  measured and kept opt-in: bge-reranker-base over the windows of the fused
  top sessions falls from 0.920 to 0.780 hit@1 on the first hundred
  questions.
- `examples/locomo_dump` and `scripts/longmemeval_qa.py --bench locomo`:
  the same retrieval dump and answer-accuracy seam for LoCoMo.

## 0.5.0 (2026-09-12)

- LongMemEval_S with an encoder: the fused panel over session documents
  reaches 0.920 hit@1 and 0.968 recall@5 on the first hundred questions,
  against 0.840 and 0.904 for the lexical ballot alone.
- `examples/longmemeval` writes the sessions each arm retrieved to
  `PACKSET_LME_DUMP`, and `scripts/longmemeval_qa.py` turns that into
  LongMemEval answer accuracy with the benchmark's reading and judge
  prompts over any OpenAI-compatible reader and judge.
- The writer reads `PACKSET_FUSE`, `PACKSET_DIVERSIFY` and `PACKSET_DECAY`
  when the flag of the same name is absent. It read only the flags, so a
  writer started with `PACKSET_DECAY=fsrs` in its environment ran with
  decay off and `status` said so.
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
