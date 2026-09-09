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
- Search asks two scorers and fuses their answers. One finds an atom
  through a typo or a prefix and weighs every word alike; BM25 weighs a
  word by how much it narrows the pack down, normalises for length, and
  finds nothing a typo hides. The milli projection is a third when it is
  built. Fusing measures better than either alone: see Retrieval below.
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
| both, Borda then MMR | **0.547** | **0.611** |

Three things this does not say. It measures the scorer, not the system. Turns
are loaded as atoms and nothing in packset extracts. So this is the ranking
given a corpus, not a judgement about what should have been remembered.
It is recall of labelled evidence with no model in the loop, so it is not the
92.5 and 94.4 the memory papers report for end-to-end answers. And it is not a
good number. Published lexical-plus-dense fusion on this benchmark reaches
Hit@1 0.752 at session granularity where the best arm here reaches 0.589. The
difference is the dense half, which packset does not have.

Read a number against its unit. The table above ranks turns. The retrieval
papers score a session by its best turn instead, which gives 0.928 hit@10 for
BM25 alone here. A session holds dozens of turns, so it is an easier question.
Fusing wins on turns and loses on sessions. Turns are what a pack stores, which
is why the panel stays the default.

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
