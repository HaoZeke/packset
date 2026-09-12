<p align="center">
  <img src="docs/logo/icon.svg" width="120" height="120" alt="packset">
</p>

# packset

**One writer. Typed atoms. Every agent is a client.**

`packsetd` is the seat-pack daemon on `127.0.0.1:8761`. Cards
(`USER.md`, `MEMORY.md`) stay files. Atoms live in LMDB. milli is
a search projection, not a Meilisearch server. Isolated harness
homes do not get a private store.

```
packset ensure          # start packsetd; print PACKSET_URL
packset pin NAME        # scope retrieve and Remember
packset atoms --as-of 2024-06-01T00:00:00.000Z
# In chat:  Remember: always open review links after pushing
```

`PACKSET_URL` is the contract. `INSIDE_MEMORY_URL` is an alias.
Default writer URL is `http://127.0.0.1:8761`.

```
packset status          # counts by kind, the pinned set, the index
packset which           # the packsetd this seat would start
packset stop
```

## Citations

An entity that opens `deed-` or `sha256:` names a product in a
[deedar](https://github.com/indynull/deedar) store, and the writer checks that
shape rather than leaving a typo for a reader to find. Every other entity is a
free-form name and is untouched.

```
packset accessions WORKSPACE           # every accession cited by a live atom
packset accessions WORKSPACE | deedar evidence -   # the bytes are intact
packset accessions WORKSPACE | deedar current -    # still the tip
packset citers ACCESSION               # the live atoms citing one accession
```

`accessions` and `citers` are the two directions of one join. A tracker answers
which issues cite a product; `citers` answers which remembered claims do.
Neither store opens the other, so what composes them is a caller holding one
accession.

Whether the deed exists is a question the pack cannot ask. It checks the shape;
`deedar` answers the rest.

A retraction cites a deed too. `POST /v1/atoms/delete` takes an optional `why`,
which has to be an accession rather than free text, and writes it onto the
tombstone beside the text it withdraws. The tombstone leaves the live set, so
`accessions` and `citers` stop reporting it, which is the point: a withdrawn
claim should not keep asserting anything. Read it back through the window it was
live in.

```
packset atoms --as-of 2026-09-11T00:00:00Z WORKSPACE   # what the pack held then
```

## Trust

A `trust` atom is one row of an influence graph: `from`, `to`, and a `weight`
in `(0, 1]`, with the usual `text` and any deeds it cites as `entities`. It
is memory, so it has a validity window and can be superseded, and `export`
carries it with the rest. The seat reads the live rows into a consensus; the
pack does not settle anything itself.

## Islands

`packset islands WS` lists the link graph's communities, largest first,
by label propagation (doi:10.1103/PhysRevE.76.036106). `packset island CUE`
returns the memories a task activates: the top five hits seed a two-hop
spread along the links, half lost per hop and divided by fan-out
(spreading activation, doi:10.1037/0033-295X.82.6.407). `GET /v1/islands`
and `GET /v1/activate` are the endpoints; the seat's `ljos island` reads
the second.

Links carry weights, and use moves them. `link_weights` sits beside `links`
on the atom, absent meaning 0.5, so every pack written so far reads
unchanged. `packset fire ID ID...` (or `island --fire`, which fires the top
eight it returns) says these claims fired together: each pair moves toward
one by a tenth of the gap, a pair with no link gains one when both have
room, and every other link of a fired claim loses two percent, the
forgetting term Oja adds to Hebb (doi:10.1007/BF00275687). Activation then
spreads in proportion to weight, so the paths a seat uses carry more and the
ones it does not fade.

## Forgetting

Every claim carries a review clock: `due_at`, and a `review` block with
stability and difficulty that `POST /v1/grade` moves (recalled grows
stability by how overdue the claim was; lapsed halves it). A new claim is due
after one day. `PACKSET_DECAY=fsrs` makes the clock a voter: a fused score is
scaled by the FSRS-4.5 retrievability `R = (1 + 19/81 * t/S)^(-1/2)`, with
`t` the days since the last review (or the write) and `S` the stability, so
`R(S) = 0.9` (doi:10.1145/3534678.3539081). Cards stay at one, and `R` floors
at 0.25 so a forgotten claim is still found when nothing else answers. The
power-law form follows Wixted and Ebbesen (doi:10.1111/j.1467-9280.1991.tb00175.x);
the spacing effect it schedules for is reviewed in Cepeda et al.
(doi:10.1037/0033-2909.132.3.354). Default off: the LoCoMo corpus has no
review history, so the slot cannot be measured there.

## Law

- One writer. Working tree is not the pack.
- `Remember:` / `Prefer:` are instant. One claim per atom.
- Search asks several scorers and fuses their answers. One finds an atom
  through a typo or a prefix and weighs every word alike; BM25 weighs a
  word by how much it narrows the pack down, normalises for length, and
  finds nothing a typo hides; the dense projection matches meaning and
  needs no shared word at all. milli is a fourth when it is built. Every
  projection is optional and its absence is a supported state. Fusing
  measures better than any alone: see Retrieval below.
- Search merge is a host voter panel. Default is CombMNZ then MMR,
  decay off, and the default was measured: see Retrieval below. `PACKSET_FUSE`, `PACKSET_DIVERSIFY`, and
  `PACKSET_DECAY` select the sequence. Not a client header. The measured
  cross-encoder second stage is off unless `PACKSET_RERANK=1` or
  `/v1/search?rerank=1`.
- Tool dumps and fetched bodies are not atoms.

## Retrieval

The fusion is measured rather than argued. `examples/locomo` runs the scorers
against LoCoMo (DOI 10.48550/arXiv.2402.17753), which labels each of its
questions with the dialogue turns that answer it:

```console
$ curl -sSLO https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
$ cargo run --release -p packset-daemon --example locomo -- locomo10.json
```

Ten conversations, 5882 turns loaded as atoms, 1536 answerable questions:

| arm | R@10 | hit@10 |
|---|---|---|
| the pack's own scorer | 0.514 | 0.580 |
| BM25 | 0.530 | 0.588 |
| the pack's scorer + BM25 | 0.547 | 0.611 |
| BM25 + dense, bge-small | 0.628 | 0.701 |
| BM25 + dense, bge-large | 0.685 | 0.755 |
| BM25 + dense, multilingual-e5-large | **0.705** | **0.781** |

### Which formula

BM25+ (Lv and Zhai, DOI 10.1145/2063576.2063584) holds each occurrence above a
floor that plain BM25's length normalisation drives toward zero. It is the
default and leads at every granularity measured; query likelihood with a
Dirichlet prior (DOI 10.1145/984321.984322) loses, alone and fused.

| scorer over passages | hit@1 | nDCG@5 |
|---|---|---|
| BM25, k1=1.2 b=0.75 | 0.660 | 0.751 |
| **BM25+**, lower-bounded TF | **0.668** | **0.753** |
| query likelihood, Dirichlet | 0.637 | 0.734 |
| BM25+ and Dirichlet fused | 0.665 | 0.751 |

### What a document is

Passage-level evidence (Callan, DOI 10.1007/978-1-4471-2099-5_31): windows of
six turns at stride three, each session scored by its best window. The size is
set from what a passage is for, not searched over. Every collapsing arm is
retrieved deep enough to fill the deepest cut-off, so the table compares
protocols, not ranking depth.

| protocol | hit@1 | nDCG@5 |
|---|---|---|
| turn ranking read as sessions | 0.615 | 0.716 |
| session as one document | 0.633 | 0.735 |
| **passage windows** | **0.660** | **0.751** |

## Which voter fuses them

The panel's default was argued and is now measured. Nine voters over the three
ballots this crate ships, same questions, same corpus, read as sessions:

| voter | hit@1 | nDCG@5 |
|---|---|---|
| CombMNZ | **0.638** | **0.722** |
| CombSUM | 0.632 | 0.721 |
| Borda | 0.620 | 0.712 |
| Dowdall | 0.618 | 0.713 |
| RRF | 0.613 | 0.708 |

The margin grows with the strength of the ballots. On session BM25 fused with
e5-large, CombMNZ leads Borda by 3.2 points of hit@1 and 0.020 nDCG@5.

Score fusion beats rank fusion here, which is what the published system on this
benchmark credits its own gain to. CombMNZ is the default.

Sweep the shipped ballots, not the strongest pair available: RRF led a pair of
two ballots by 2 points and trailed Borda on the three the seat actually fuses.
A default chosen from the first would have been a claim about a retriever
nobody runs.

Ranked pairs is the one voter that says little here. It decides a pair by which
majority prefers it, and two voters disagreeing is one against one, so few
victories lock and the order falls back to the first ballot. It is for a panel
of three or more.

### Which diversifier

No difference, so the default stays. A diversifier suppresses redundancy and
this benchmark scores recall of labelled evidence, so it cannot see what the
slot is for. `PACKSET_DIVERSIFY` picks; DPP is greedy MAP over a
quality-diversity kernel (DOI 10.1561/2200000044).

| diversify | hit@1 | nDCG@5 |
|---|---|---|
| `mmr` (shipped) | 0.732 | 0.809 |
| `dpp` | 0.732 | 0.810 |
| `none` | 0.732 | 0.810 |

### A second stage

A cross-encoder reads question and candidate together (monoBERT, DOI
10.48550/arXiv.1901.04085); `PACKSET_LOCOMO_RERANK=1` reorders the top 20 of
the fused list with `packset-embed --rerank`. Against the free change it
loses: score-level fusion with no model matches its hit@1 and beats it
elsewhere, at a forward pass per candidate per question less. Off by default.

| over passage BM25+ + dense, Borda | hit@1 | hit@5 | nDCG@5 |
|---|---|---|---|
| first stage | 0.711 | 0.922 | 0.794 |
| reranked, bge-reranker-base | 0.730 | 0.913 | 0.797 |
| first stage, CombSUM instead of Borda | **0.732** | **0.933** | **0.809** |

## What a question costs

`cargo bench -p packset-core` times the index build, one question per scorer,
and the fuse. "Before" built a hit for every candidate and sorted them all; a
bounded heap now keeps the k best. The pack's own scan scores by the tokens
the index was built from rather than re-tokenising: 60 ms to 13 ms at 10k
atoms in one run. Fusing two ballots of twenty costs 79 µs bare, 655 µs under
DPP, 917 µs under MMR.

| atoms | index build | BM25+ question, before | after | the pack's own scan |
|---|---|---|---|---|
| 1,000 | 1.2 ms | 1.7 ms | **0.23 ms** | 3.8 ms |
| 10,000 | 13 ms | 41 ms | **2.3 ms** | 55 ms |
| 100,000 | 95 ms | 438 ms | **53 ms** | 795 ms |

## LongMemEval

Session retrieval on LongMemEval_S (doi:10.48550/arXiv.2410.10813): 500
questions, about fifty chat sessions each, the answer sessions labelled; the
30 abstention questions are excluded. Turns loaded as atoms, BM25+ only, no
model in the loop. `hit@k` is any answer session in the top k; `recall@k`
is the fraction of answer sessions there. `cargo run --release -p
packset-daemon --example longmemeval -- longmemeval_s.json`, 44 s for 470
questions.

| arm | hit@1 | recall@5 | recall@10 |
|---|---|---|---|
| turns, read as sessions | 0.864 | 0.906 | 0.948 |
| passage windows of six turns | 0.851 | 0.907 | 0.951 |
| session documents | 0.855 | 0.914 | 0.952 |

By question type the protocols agree within two points except
`single-session-preference` (30 questions), where turns reach 0.300 hit@1
and windows or sessions 0.467: a preference is stated across a session, not
in one turn, and it is the type a lexical scorer serves worst. Multi-session
questions reach 0.87 recall@5 under every protocol. The dense and fused
arms on this benchmark are the next row.

## Many clients on one writer

`cargo run --release -p packset-daemon --example hammer -- CLIENTS OPS`
runs that many clients against one writer, each remembering unique claims
and searching twice per claim. Four workers on four cores, 100 operations a
client, 300 requests each:

| clients | encoder present, req/s | remember p50 / p99 | search p50 / p99 | encoder absent, req/s | remember p50 / p99 | search p50 / p99 |
|---|---|---|---|---|---|---|
| 1 | 45 | 12 ms / 0.9 s | 4.5 ms / 6.5 ms | 42 | 10 ms / 71 ms | 4.9 ms / 19 ms |
| 4 | 145 | 29 ms / 0.4 s | 11 ms / 26 ms | 228 | 15 ms / 117 ms | 16 ms / 25 ms |
| 16 | 177 | 62 ms / 1.1 s | 65 ms / 0.2 s | 220 | 60 ms / 0.4 s | 65 ms / 89 ms |
| 32 | 147 | 158 ms / 2.3 s | 154 ms / 2.0 s | 181 | 144 ms / 1.9 s | 138 ms / 1.5 s |

No request failed. One logical write runs at a time and the encoder runs
before that lock; with the encoder present every remember and search pays
one encode, which is where the two columns part. Latency past four clients
is queueing on four workers: `PACKSET_WORKERS` sets the pool. A second
host is a client over the network or its own pack, not a second writer.

## What did not work

Pseudo-relevance feedback, and the way it fails is the useful part.

`Index::expand` implements RM3: estimate the rest of the query from the
documents the first pass returned, keeping half the weight on the words
actually asked for (Lavrenko and Croft, `10.1145/383952.383972`). Nothing is
trained and nothing is stored.

| corpus | BM25 | with RM3 |
|---|---|---|
| turns as documents, hit@1 | 0.589 | 0.451 |
| sessions as documents, hit@1 | 0.607 | 0.604 |

On dialogue turns it costs 14 points. On the same corpus indexed as sessions it
costs nothing. A relevance model is a language model estimated from the top of
the first pass, and a turn is ten to thirty tokens: ten of them is not enough
text to estimate from, and with hit@1 near 0.29 most of what it estimates from
is wrong. The model then reaches for more documents like the wrong ones.

So it stays implemented, tested and off. It is a bad trade for short atoms,
which is what a pack holds, and the measurement says why rather than that.

Two things this does not say. It measures the scorer, not the system. Turns are
loaded as atoms and nothing in packset extracts. So this is the ranking given a
corpus, not a judgement about what should have been remembered. And it is
recall of labelled evidence with no model in the loop, so it is not the 92.5 and
94.4 the memory papers report for end-to-end answers.

Read a number against its unit, and read the unit twice. The table above ranks
turns. The retrieval papers score a session instead, and "at session
granularity" covers two protocols that are not the same retriever: rank the
turns and read off which session each came from, so a session is scored by its
single best turn, or index the session itself, so BM25 gets one long document
and a question whose words are spread over several turns matches what no single
turn matches.

Indexing the session is worth 1.8 points of hit@1 over reading sessions off a
turn ranking, with no vectors and no model, and passage windows are worth 4.5
over the same baseline. A published BM25 baseline several points above one
measured here is more likely a different unit than a better implementation of
the same formula, and `examples/locomo` reports every protocol side by side so
the question is answerable rather than arguable.

## Where this sits against the published numbers

The best arm measured here, against the best published on this benchmark:

| | session hit@1 | nDCG@5 |
|---|---|---|
| what this shipped before | 0.549 | 0.660 |
| session BM25, no stemming | 0.607 | 0.710 |
| session BM25, stemmed | 0.633 | 0.735 |
| passage BM25+ | 0.668 | 0.753 |
| session BM25 + dense, CombMNZ | 0.722 | 0.802 |
| passage BM25+ + dense (multilingual-e5-large), CombSUM | 0.732 | 0.809 |
| passage BM25+ + dense (e5-large-v2), CombSUM | **0.736** | **0.812** |
| published, BM25 + e5-large-v2 | 0.752 | 0.829 |

Same ten conversations, same 1536 questions, and every method here is
training-free. Not the same encoder: what `PACKSET_EMBED_MODEL=e5-large`
loads is `multilingual-e5-large`, and the paper used the English `e5-large-v2`,
which the model runtime does not carry. They are different models with
different training data and the multilingual one scores lower on English
retrieval. Every row above that says "dense" was measured with the multilingual
one, and this table said "e5-large-v2" for it until that was checked. The
English model is now loadable from files as `e5-large-v2`, and with it the
best arm is 0.736 hit@1 and 0.812 nDCG@5: the encoder accounted for 0.004 of
the residual, not the residual. The remaining 0.016 hit@1 is the sample or
the BM25 side, as the section below says.

### Which encoder, and whether it has to be dense

SPLADE++ (DOI 10.1145/3404835.3463098) is a peer of BM25+ alone and matches
e5-large-v2 dense as a fusion partner, from a 110M model whose output lives in
an inverted index. The passage protocol is a lexical gain: passage dense
scores no better than turn dense. gte-large and mxbai-large lose by a distance
the leaderboards do not predict; the runtime's gte export may not be the
reference model.

| ballot | hit@1 | nDCG@5 |
|---|---|---|
| BM25+ | 0.635 | 0.733 |
| **SPLADE++**, learned sparse (`--sparse`) | **0.637** | **0.742** |
| dense, e5-large-v2 | 0.667 | 0.765 |
| dense, multilingual-e5-large | 0.653 | 0.753 |
| dense, mxbai-embed-large-v1 | 0.562 | 0.689 |
| dense, gte-large-en-v1.5 | 0.477 | 0.607 |
| BGE-M3 sparse head | 0.553 | |
| pair | hit@1 | nDCG@5 |
|---|---|---|
| + dense, e5-large-v2 | 0.711 | 0.794 |
| + **SPLADE++** | **0.714** | 0.787 |
| + passage dense, e5-large-v2 | 0.709 | 0.791 |
| session granularity, ten conversations | hit@1 | nDCG@5 |
|---|---|---|
| session BM25 + dense (multilingual-e5-large), CombMNZ | **0.716** | **0.794** |
| session BM25 + per-token (BGE-M3 int8), CombMNZ | 0.673 | 0.763 |
| session BM25 + learned sparse (BGE-M3), Borda | 0.629 | 0.716 |
| arm | R@10 | session hit@1 |
|---|---|---|
| its pooled vector, by cosine | 0.562 | 0.499 |
| its per-token vectors, by max-sim | **0.649** | **0.564** |
| BM25 + pooled | 0.635 | 0.590 |
| BM25 + max-sim | **0.653** | **0.642** |

## Crates

| Crate | Role |
|---|---|
| `packset-core` | atom schema, prose and readability, recall, search scoring, the named fuse/diversify panel, decay, extract filters |
| `packset-daemon` | the writer: the LMDB store, the cards, and `/v1` |
| `packset-client` | HTTP client |
| `packset-cli` | `packset`: lifecycle and the `/v1` reads a shell runs |
| `packset-milli` | inverted-index projection (build on the remote builder) |
| `packset-embed` | dense projection: text in, vectors out (build where the model is) |

One process owns `memory.lmdb`; clients never open it. `packset` finds the
`packsetd` beside itself, so a checkout runs its own build rather than
whichever one is on `PATH`, and `packset which` says which that is.

`crates/packset-core/tests/goldens.json` fixes the accept and reject
boundary: which records the writer stores, which it refuses, and the exact
words it refuses them with. It is checked-in data rather than generated
output, so a diff there is a change in what the daemon accepts.

## Clients

Harness launchers and the seated agent talk HTTP. They do not own
the store.

## Build

Do not compile on a laptop. `just milli` and `cargo test` run on
the remote builder.

## License

MIT
