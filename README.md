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
  `PACKSET_DECAY` select the sequence. Not a client header.
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
| BM25 + dense, e5-large-v2 | **0.705** | **0.781** |

`PACKSET_EMBED_MODEL` picks the encoder. bge-small is the default because it is
130 MB against 1.3 GB and encodes about three times faster, and it costs 0.08
R@10.

Size is part of it and not all of it. bge-base measures no better than
bge-small, so the small end is not a ladder; bge-large is worth 0.06 R@10 over
bge-small. But e5-large-v2 is the same 335M parameters as bge-large and measures
0.02 R@10 above it, which is a third of what the whole step up from small bought.
Which family was trained how matters at the top end, so a seat naming a larger
model should name which larger model.

Every ballot is optional. A seat with no encoder gets the first three rows and
loses 0.08 to 0.14 R@10, which is what the dense projection is worth.

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

Indexing the session is worth 1.8 points of hit@1 and 5.7 of R@20 over reading
sessions off a turn ranking, with no vectors and no model. A published BM25
baseline several points above one measured here is more likely a different unit
than a better implementation of the same formula, and `examples/locomo` reports
both protocols side by side so the question is answerable rather than arguable.

## Where this sits against the published numbers

The best arm measured here, against the best published on this benchmark:

| | session hit@1 | nDCG@5 |
|---|---|---|
| what this shipped before | 0.549 | 0.660 |
| session BM25 alone | 0.607 | 0.710 |
| session BM25 + dense, Borda | 0.684 | 0.774 |
| session BM25 + dense, CombMNZ | **0.716** | **0.794** |
| published, BM25 + e5-large-v2 | 0.752 | 0.829 |

Same encoder family as the published system, same ten conversations, same 1536
questions, and every method here is training-free.

The gain reproduces almost exactly. That paper reports +11.2 points over BM25
alone; fusing dense into session BM25 here is worth +10.9, and CombMNZ over
Borda is 3.2 of those. What does not reproduce is the starting point: their
BM25 baseline implies 0.640 against the 0.607 measured here, and 0.033 of
baseline accounts for almost all of the 0.036 that remains.

So the residual is a difference in the BM25 side or in the sample, not in the
fusion. The paper does not state which subset of LoCoMo it used and the family
runs to fifty dialogues where the public file holds ten, so that last part is
not closable from here and is not worth tuning against.

Of the distance that did close, from 0.203 to 0.036, three of four causes were
defects rather than missing capability. Score fusion parsed and never ran. A
relevance model weighted rarity twice. A voter could stop the process. The
fourth was a unit: "session granularity" naming two protocols.

What has not been tried at this scale is late interaction on the session
protocol, which led on a three-conversation subset. It needs one vector per
token rather than one per atom.

Late interaction is the difference, and it is the scoring rather than the model.
One model scored both ways, BGE-M3 over three of the conversations:

| arm | R@10 | session hit@1 |
|---|---|---|
| its pooled vector, by cosine | 0.562 | 0.499 |
| its per-token vectors, by max-sim | **0.649** | **0.564** |
| BM25 + pooled | 0.635 | 0.590 |
| BM25 + max-sim | **0.653** | **0.642** |

The same weights, the same corpus, the same questions. Scoring a document by the
best match each query token finds anywhere in it beats pooling those tokens into
one vector, by 0.087 R@10 and 0.065 session hit@1 alone, and by 0.052 session
hit@1 once BM25 is fused in.

Size is not hiding in there either: BGE-M3's pooled output at 568M parameters is
*worse* here than bge-small's at 33M, 0.562 R@10 against 0.641.

`packset-embed --late` and `search::max_sim` implement it, and nothing in the
writer reads them. A vector per token is thirty vectors where the pooled form is
one, so an atom's `embedding` would grow accordingly, and whether a pack can
afford that depends on how large a pack is.

The stored link graph does not help a query. Given twenty places, filling the
last ten by following the neighbours of the first ten scores 0.562 R@20 against
0.617 for letting the ranking continue. A neighbourhood is for walking out from
something already found, not for answering.

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
