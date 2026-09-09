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
- Search merge is a host voter panel. Default is Borda then MMR,
  decay off. `PACKSET_FUSE`, `PACKSET_DIVERSIFY`, and
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
| dense (bge-small-en-v1.5) | 0.585 | 0.656 |
| the pack's scorer + BM25 | 0.547 | 0.611 |
| all three | 0.619 | 0.690 |
| BM25 + dense | **0.628** | **0.701** |

Every ballot is optional. A seat with no encoder gets the first two rows and
loses about 0.08 R@10, which is what the dense projection is worth.

Two things this does not say. It measures the scorer, not the system. Turns are
loaded as atoms and nothing in packset extracts. So this is the ranking given a
corpus, not a judgement about what should have been remembered. And it is
recall of labelled evidence with no model in the loop, so it is not the 92.5 and
94.4 the memory papers report for end-to-end answers.

Read a number against its unit. The table above ranks turns. The retrieval
papers score a session by its best turn instead. Read that way, BM25 with dense
reaches hit@1 0.612 and hit@10 0.951, against 0.752 hit@1 published for
lexical fused with a dense retriever ten times the size of this one.

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
