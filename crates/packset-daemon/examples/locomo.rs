//! Retrieval quality, against somebody else's relevance judgements.
//!
//! Every claim this crate makes about search has been about mechanism and cost.
//! Neither says whether an answer is any good, and the two scorers were fused
//! on the argument that they are strong at different queries, which is a
//! hypothesis rather than a measurement.
//!
//! LoCoMo (DOI 10.48550/arXiv.2402.17753) supplies the judgements: 1986
//! questions over ten long conversations, each question labelled with the
//! dialogue turns that answer it. That is a retrieval ground truth, and it does
//! not care who wrote the retriever.
//!
//! ```console
//! $ curl -sSLO https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
//! $ cargo run --release -p packset-daemon --example locomo -- locomo10.json
//! ```
//!
//! What this measures and what it does not:
//!
//! - It measures the scorer. Turns are loaded as atoms, which is not how a pack
//!   is built: nothing in packset extracts, and an atom exists because a person
//!   wrote one. Giving every arm the same corpus is what isolates the ranking
//!   from the question of what should have been remembered.
//! - It is not the number the memory papers report. Those are end-to-end answer
//!   accuracy with a model reading the retrieved context and a model judging the
//!   answer. This is recall of labelled evidence, with no model in the loop, so
//!   it is not comparable to 92.5 or 94.4 and is not offered as though it were.
//! - Category 5 is adversarial, meaning the conversation does not answer the
//!   question. Recall is undefined there and those questions are excluded.
//!
//! Four knobs, all off by default and all about what a run costs:
//! `PACKSET_LOCOMO_LATE` adds the per-token arms, `PACKSET_LOCOMO_WALK` adds
//! the restart walk over the link graph, `PACKSET_LOCOMO_CONVERSATIONS` scores
//! the first N, and `PACKSET_LOCOMO_CACHE` names a directory to keep the
//! encodings in. A capped run keeps the arms comparable to each other and stops
//! them being comparable to a run over all ten, so a number from one says which
//! it was.

use std::collections::BTreeSet;

use packset_core::bm25::Index;
use packset_core::panel::Panel;
use packset_core::search::{self, Ask, Record};
use serde_json::{json, Value};

/// Cut-offs to report. A pack hands a model a small context, so the small ones
/// are the ones that matter.
const CUTOFFS: &[usize] = &[1, 5, 10, 20];

/// LoCoMo's adversarial category: the conversation does not answer it.
const ADVERSARIAL: i64 = 5;

/// One question and the turns that answer it.
struct Question {
    text: String,
    evidence: BTreeSet<String>,
    category: i64,
}

/// One conversation: its turns as atoms, and its questions.
struct Conversation {
    atoms: Vec<Record>,
    questions: Vec<Question>,
}

/// What one arm scored, summed over questions.
#[derive(Default, Clone)]
struct Tally {
    /// Fraction of labelled evidence inside the cut-off, summed.
    recall: Vec<f64>,
    /// Questions with at least one labelled turn inside the cut-off.
    hit: Vec<usize>,
    /// Discounted gain at `NDCG_CUT`, summed, against the ideal ordering.
    ndcg: f64,
    /// Questions counted.
    asked: usize,
}

/// Where NDCG is reported. The published number on this benchmark is at five.
const NDCG_CUT: usize = 5;

/// Discounted cumulative gain of a ranking, against the best it could be.
///
/// Relevance is binary here, because the benchmark labels a turn as evidence
/// or not and says nothing about how much. So the ideal ranking puts every
/// labelled item first, and the ideal gain depends only on how many there are.
fn ndcg_at(ranked: &[String], evidence: &BTreeSet<String>, cut: usize) -> f64 {
    let discount = |place: usize| 1.0 / ((place + 2) as f64).log2();
    let gain: f64 = ranked
        .iter()
        .take(cut)
        .enumerate()
        .filter(|(_, id)| evidence.contains(*id))
        .map(|(place, _)| discount(place))
        .sum();
    let ideal: f64 = (0..cut.min(evidence.len())).map(discount).sum();
    if ideal <= 0.0 {
        0.0
    } else {
        gain / ideal
    }
}

impl Tally {
    fn new() -> Self {
        Self {
            recall: vec![0.0; CUTOFFS.len()],
            hit: vec![0; CUTOFFS.len()],
            ndcg: 0.0,
            asked: 0,
        }
    }

    fn add(&mut self, ranked: &[String], evidence: &BTreeSet<String>) {
        self.asked += 1;
        self.ndcg += ndcg_at(ranked, evidence, NDCG_CUT);
        for (slot, cut) in CUTOFFS.iter().enumerate() {
            let seen: BTreeSet<&String> = ranked.iter().take(*cut).collect();
            let found = evidence.iter().filter(|id| seen.contains(id)).count();
            self.recall[slot] += found as f64 / evidence.len() as f64;
            if found > 0 {
                self.hit[slot] += 1;
            }
        }
    }

    fn merge(&mut self, other: &Self) {
        self.asked += other.asked;
        self.ndcg += other.ndcg;
        for slot in 0..CUTOFFS.len() {
            self.recall[slot] += other.recall[slot];
            self.hit[slot] += other.hit[slot];
        }
    }
}

/// The id a hit names, which for these atoms is the dialogue turn.
fn hit_ids(hits: &[Value]) -> Vec<String> {
    hits.iter()
        .filter_map(|hit| hit.get("id").and_then(Value::as_str).map(str::to_string))
        .collect()
}

fn conversations(raw: &Value) -> Vec<Conversation> {
    let mut out = Vec::new();
    for sample in raw.as_array().map(Vec::as_slice).unwrap_or_default() {
        let mut atoms: Vec<Record> = Vec::new();
        let Some(conversation) = sample.get("conversation").and_then(Value::as_object) else {
            continue;
        };
        // Sessions are numbered keys beside their date, so take the arrays.
        let mut sessions: Vec<(&String, &Value)> = conversation
            .iter()
            .filter(|(key, value)| key.starts_with("session_") && value.is_array())
            .collect();
        sessions.sort_by_key(|(key, _)| {
            key.trim_start_matches("session_")
                .parse::<u32>()
                .unwrap_or(u32::MAX)
        });
        for (_, turns) in sessions {
            for turn in turns.as_array().map(Vec::as_slice).unwrap_or_default() {
                let (Some(id), Some(text)) = (
                    turn.get("dia_id").and_then(Value::as_str),
                    turn.get("text").and_then(Value::as_str),
                ) else {
                    continue;
                };
                let speaker = turn
                    .get("speaker")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                // No `entities` field, so the extractor runs the way it does
                // for an atom nobody annotated: proper nouns out of the text.
                // Naming the speaker as the only entity would make every pair
                // of turns by one person identical to the link rule, and the
                // graph arm below would be measuring nothing.
                let atom = json!({
                    "id": id,
                    "workspace": "locomo",
                    "kind": "conclusion",
                    "text": format!("{speaker}: {text}"),
                });
                atoms.push(atom.as_object().expect("object").clone());
            }
        }

        let questions = sample
            .get("qa")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .filter_map(|qa| {
                let text = qa.get("question").and_then(Value::as_str)?.to_string();
                let evidence: BTreeSet<String> = qa
                    .get("evidence")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect();
                Some(Question {
                    text,
                    evidence,
                    category: qa.get("category").and_then(Value::as_i64).unwrap_or(0),
                })
            })
            .collect();
        out.push(Conversation { atoms, questions });
    }
    out
}

/// The arms compared at session granularity, where the difference between them
/// is what a document is.
///
/// A retrieval number reported "at session granularity" can mean either of two
/// protocols, and they are not the same retriever. Rank the turns and read off
/// which session each came from, so a session is scored by its single best
/// turn; or index the session itself, so every word anyone said in it counts
/// toward one score. A published BM25 baseline five points above the one here
/// is more likely a different unit than a better implementation of the same
/// formula, and this table is what says which.
const PROTOCOLS: &[&str] = &[
    "turn bm25",
    "session bm25",
    "session bm25 rm3",
    "turn dense",
    "session bm25 + turn dense",
    "session bm25 + turn late",
    "session bm25 + turn m3 sparse",
];

/// The arms compared, each a way of turning one question into a ranking.
const ARMS: &[&str] = &[
    "lexical",
    "bm25",
    "bm25 rm3",
    "dense",
    "lexical+bm25 (shipped)",
    "lexical+bm25+dense",
    "bm25+dense",
    "bm25 rm3+dense",
    "late",
    "bm25+late",
    "m3 pooled",
    "bm25+m3 pooled",
    "m3 sparse",
    "bm25+m3 sparse",
    "m3 sparse+late",
];

/// Every fusion the panel accepts, run over one pair of ballots.
///
/// The published lexical-plus-dense system on this benchmark attributes its
/// gain to fusing at the score level rather than the rank level, and packset
/// ships a rank fusion. Fusing ballots that are already computed costs a merge,
/// so the question is answered inside the run that produced them rather than by
/// nine runs of the encoder.
const VOTERS: &[&str] = &[
    "borda", "rrf", "combsum", "combmnz", "dowdall", "kemeny", "schulze", "copeland", "tideman",
];

/// The voters swept a second time, over the ballots the seat ships.
///
/// Five rather than nine, because the second sweep is what takes a run past
/// what this builder lets finish. Schulze, ranked pairs and Copeland build a
/// pairwise matrix over every candidate, which is the expensive part, and the
/// sweep above already measures all three: the first two degenerate to their
/// own first ballot on two voters, and Copeland tracks Borda. That every name
/// reaches its own implementation is a property, and it is pinned by a test
/// rather than by paying for it once a question here.
const SHIPPED_VOTERS: &[&str] = &["borda", "rrf", "combsum", "combmnz", "dowdall"];

/// Where encodings are kept between runs, when the seat names a directory.
///
/// The encode dominates a run, costs the same every time, and depends on
/// nothing but the model and the text. A benchmark that pays it again on every
/// run is a benchmark that cannot be re-run, which is how a question about
/// fusion ends up waiting on a question about a scheduler.
fn cache_dir() -> Option<std::path::PathBuf> {
    let raw = std::env::var_os("PACKSET_LOCOMO_CACHE")?;
    let dir = std::path::PathBuf::from(raw);
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The model the vectors came from, so two models never share a file.
fn model_name() -> String {
    std::env::var("PACKSET_EMBED_MODEL").unwrap_or_else(|_| "default".to_string())
}

/// Rows of floats, length-prefixed, so a ragged set reads back as it was
/// written and a truncated file fails the count check rather than the scoring.
fn write_rows(path: &std::path::Path, rows: &[Vec<f32>]) {
    // A run whose encoder died halfway has rows that are empty rather than
    // wrong, and caching those would make the next run read a failure as an
    // answer. An absent encoder is a supported state; a cached absence is not.
    if rows.is_empty() || rows.iter().any(Vec::is_empty) {
        return;
    }
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for row in rows {
        bytes.extend_from_slice(&(row.len() as u32).to_le_bytes());
        for value in row {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    // Written beside and renamed, so a cancelled run leaves no half file that
    // the next one would read as complete.
    let temporary = path.with_extension("part");
    if std::fs::write(&temporary, &bytes).is_ok() {
        let _ = std::fs::rename(&temporary, path);
    }
}

fn read_rows(path: &std::path::Path, expected: usize) -> Option<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path).ok()?;
    let mut at = 0usize;
    let mut take = |width: usize| -> Option<&[u8]> {
        let slice = bytes.get(at..at + width)?;
        at += width;
        Some(slice)
    };
    let count = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    if count != expected {
        return None;
    }
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count {
        let width = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let raw = take(width * 4)?;
        rows.push(
            raw.as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes)
                .collect(),
        );
    }
    Some(rows)
}

/// How many conversations to score, when a machine cannot hold a whole run.
///
/// All ten by default. A smaller number is not a smaller benchmark so much as a
/// different one, and the arms stay comparable to each other because they all
/// see the same corpus; they stop being comparable to a run over ten.
fn conversation_cap() -> Option<usize> {
    std::env::var("PACKSET_LOCOMO_CONVERSATIONS")
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .filter(|n| *n > 0)
}

/// Whether to spend the time and memory on the per-token encoding.
///
/// Off unless asked, because it is one vector per token: the same corpus that
/// costs twenty five megabytes pooled costs about a gigabyte this way, and the
/// encode takes several times as long.
/// Whether to spend the time on the restart walk.
///
/// Off unless asked. The walk touches every edge on every round for every
/// question, which is most of a run's time, and it has answered its question
/// twice over: on three conversations and on ten, filling the last ten places
/// of twenty by a personalised PageRank scored below letting the ranking
/// continue, and barely above one hop. Leaving it on taxes every future run
/// with a settled question.
fn walk_wanted() -> bool {
    std::env::var("PACKSET_LOCOMO_WALK").is_ok_and(|v| !v.is_empty() && v != "0")
}

fn late_wanted() -> bool {
    std::env::var("PACKSET_LOCOMO_LATE").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Where the one-hop comparison is made: half the places are retrieved and the
/// rest are filled, either by the ranking continuing or by neighbours.
const HOP_CUT: usize = 20;

/// The session a dialogue turn belongs to, from `D<session>:<turn>`.
fn session_of(id: &str) -> &str {
    id.split_once(':').map_or(id, |(session, _)| session)
}

/// A turn ranking read as a session ranking, best turn first.
///
/// The retrieval papers on this benchmark score a session by its best turn and
/// ask whether the right session is in the top k. That is an easier question
/// than which turn, because a session holds dozens, and reporting both is what
/// keeps a number from being read against one it does not answer.
fn sessions_of(ranked: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    ranked
        .iter()
        .map(|id| session_of(id).to_string())
        .filter(|session| seen.insert(session.clone()))
        .collect()
}

/// A conversation's sessions as one document each, in the order they happened.
///
/// Concatenating a session's turns is a different document from any of them:
/// it is longer, so BM25's length normalisation treats it differently, and a
/// question whose words are spread over several turns matches it where it
/// matches no single turn.
fn session_documents(atoms: &[Record]) -> Vec<Record> {
    let mut order: Vec<String> = Vec::new();
    let mut bodies: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for atom in atoms {
        let id = atom.get("id").and_then(Value::as_str).unwrap_or_default();
        let room = session_of(id).to_string();
        let line = atom.get("text").and_then(Value::as_str).unwrap_or_default();
        match bodies.get_mut(&room) {
            Some(held) => {
                held.push('\n');
                held.push_str(line);
            }
            None => {
                order.push(room.clone());
                bodies.insert(room, line.to_string());
            }
        }
    }
    order
        .into_iter()
        .map(|room| {
            let text = bodies.remove(&room).unwrap_or_default();
            json!({
                "id": room,
                "workspace": "locomo",
                "kind": "conclusion",
                "text": text,
            })
            .as_object()
            .expect("object")
            .clone()
        })
        .collect()
}

/// A turn ranking read as a session ranking: each session keeps the place of
/// its best turn, which is the maximum-similarity protocol the papers state.
fn collapse(hits: &[Value]) -> Vec<Value> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for hit in hits {
        let id = hit.get("id").and_then(Value::as_str).unwrap_or_default();
        let room = session_of(id).to_string();
        if !seen.insert(room.clone()) {
            continue;
        }
        let mut copy = hit.clone();
        copy["id"] = json!(room);
        out.push(copy);
    }
    out
}

/// Rank the corpus by late interaction, in the same hit shape as the others.
///
/// Ordinals line up with `ask.atoms`, the way the BM25 index's do.
fn rank_late(ask: &Ask<'_>, query: &[Vec<f32>], documents: &[Vec<Vec<f32>>]) -> Vec<Value> {
    let mut hits: Vec<Value> = Vec::new();
    for (ordinal, atom) in ask.atoms.iter().enumerate() {
        let Some(tokens) = documents.get(ordinal) else {
            continue;
        };
        let relevance = search::max_sim(query, tokens);
        if relevance <= 0.0 {
            continue;
        }
        hits.push(json!({
            "field": "atom",
            "id": atom.get("id").cloned().unwrap_or(Value::Null),
            "kind": atom.get("kind").cloned().unwrap_or(Value::Null),
            "text": atom.get("text").cloned().unwrap_or(Value::Null),
            "score": relevance,
        }));
    }
    sort_by_score(&mut hits);
    hits.truncate(ask.limit);
    hits
}

/// Rank by the learned term weights the same pass returned.
///
/// The scorer is a dot product over shared vocabulary entries, so this is the
/// inverted index's shape with the weights learned rather than counted. It
/// costs one number a term where the per-token form costs a vector a token.
fn rank_sparse(
    ask: &Ask<'_>,
    query: &packset_daemon::embed::Sparse,
    documents: &[packset_daemon::embed::Sparse],
) -> Vec<Value> {
    let mut hits: Vec<Value> = Vec::new();
    for (ordinal, atom) in ask.atoms.iter().enumerate() {
        let Some(weights) = documents.get(ordinal).filter(|held| !held.is_empty()) else {
            continue;
        };
        let relevance = search::sparse_dot(query, weights);
        if relevance <= 0.0 {
            continue;
        }
        hits.push(json!({
            "field": "atom",
            "id": atom.get("id").cloned().unwrap_or(Value::Null),
            "kind": atom.get("kind").cloned().unwrap_or(Value::Null),
            "text": atom.get("text").cloned().unwrap_or(Value::Null),
            "score": relevance,
        }));
    }
    sort_by_score(&mut hits);
    hits.truncate(ask.limit);
    hits
}

/// Best score first, then the id, so a tie is broken the same way every run.
fn sort_by_score(hits: &mut [Value]) {
    hits.sort_by(|a, b| {
        b["score"]
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&a["score"].as_f64().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a["id"]
                    .as_str()
                    .unwrap_or("")
                    .cmp(b["id"].as_str().unwrap_or(""))
            })
    });
}

/// Rank by cosine against vectors held beside the atoms rather than inside
/// them, which is what the per-token encoder's pooled output needs.
fn rank_pooled(ask: &Ask<'_>, query: &[f32], documents: &[Vec<f32>]) -> Vec<Value> {
    let mut hits: Vec<Value> = Vec::new();
    for (ordinal, atom) in ask.atoms.iter().enumerate() {
        let Some(vector) = documents.get(ordinal).filter(|v| !v.is_empty()) else {
            continue;
        };
        let relevance = search::cosine(query, vector);
        if relevance <= 0.0 {
            continue;
        }
        hits.push(json!({
            "field": "atom",
            "id": atom.get("id").cloned().unwrap_or(Value::Null),
            "kind": atom.get("kind").cloned().unwrap_or(Value::Null),
            "text": atom.get("text").cloned().unwrap_or(Value::Null),
            "score": relevance,
        }));
    }
    sort_by_score(&mut hits);
    hits.truncate(ask.limit);
    hits
}

/// The stored link graph, as positions rather than ids.
///
/// Built once per conversation because a restart walk touches every edge on
/// every round, and rebuilding the adjacency per question would dominate what
/// the walk itself costs.
struct Graph {
    ids: Vec<String>,
    at: std::collections::HashMap<String, usize>,
    edges: Vec<Vec<usize>>,
}

impl Graph {
    fn of(atoms: &[Record]) -> Self {
        // One pass, so a node's position in `ids` is its position in `edges`:
        // an atom without an id would otherwise shift every edge after it.
        let held: Vec<&Record> = atoms
            .iter()
            .filter(|atom| atom.get("id").and_then(Value::as_str).is_some())
            .collect();
        let ids: Vec<String> = held
            .iter()
            .filter_map(|atom| atom.get("id").and_then(Value::as_str))
            .map(str::to_string)
            .collect();
        let at: std::collections::HashMap<String, usize> = ids
            .iter()
            .enumerate()
            .map(|(index, id)| (id.clone(), index))
            .collect();
        let edges = held
            .iter()
            .map(|atom| {
                atom.get("links")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(Value::as_str)
                    .filter_map(|peer| at.get(peer).copied())
                    .collect()
            })
            .collect();
        Self { ids, at, edges }
    }

    /// Personalised PageRank from a seeded restart distribution.
    ///
    /// One hop is the weakest thing a graph can do for a query, and measuring
    /// only that understates what a graph is worth. The retrieval work that
    /// reports a gain from a memory graph runs a restart random walk over it,
    /// which reaches a node no seed links to directly and weights it by how
    /// many seeds reach it at all. Page, Brin, Motwani, Winograd, The PageRank
    /// citation ranking, 1999; the seeded form as HippoRAG uses it,
    /// DOI 10.48550/arXiv.2405.14831.
    fn walk(&self, seeds: &[String], damping: f64, rounds: usize) -> Vec<(usize, f64)> {
        let n = self.ids.len();
        let mut restart = vec![0.0f64; n];
        let mut mass = 0.0;
        // Weighted by where the ranking put it, so the top seed pulls hardest.
        for (place, id) in seeds.iter().enumerate() {
            if let Some(index) = self.at.get(id) {
                let weight = 1.0 / (place + 1) as f64;
                restart[*index] += weight;
                mass += weight;
            }
        }
        if mass == 0.0 {
            return Vec::new();
        }
        for value in &mut restart {
            *value /= mass;
        }
        let mut rank = restart.clone();
        let mut next = vec![0.0f64; n];
        for _ in 0..rounds {
            next.iter_mut().for_each(|value| *value = 0.0);
            let mut dangling = 0.0;
            for (node, peers) in self.edges.iter().enumerate() {
                if peers.is_empty() {
                    dangling += rank[node];
                    continue;
                }
                let share = rank[node] / peers.len() as f64;
                for peer in peers {
                    next[*peer] += share;
                }
            }
            for (index, value) in next.iter_mut().enumerate() {
                // A node with no links returns its mass to the restart, which
                // keeps the walk a distribution rather than leaking it.
                *value = damping * (*value + dangling * restart[index])
                    + (1.0 - damping) * restart[index];
            }
            std::mem::swap(&mut rank, &mut next);
        }
        let mut order: Vec<(usize, f64)> = rank.into_iter().enumerate().collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        order
    }

    /// The ranking's own places kept, the rest filled by the walk.
    fn expand(&self, ranked: &[String], limit: usize) -> Vec<String> {
        let mut out: Vec<String> = ranked.to_vec();
        let mut seen: BTreeSet<&str> = ranked.iter().map(String::as_str).collect();
        for (index, score) in self.walk(ranked, WALK_DAMPING, WALK_ROUNDS) {
            if out.len() >= limit {
                break;
            }
            if score <= 0.0 {
                break;
            }
            let id = &self.ids[index];
            if seen.insert(id.as_str()) {
                out.push(id.clone());
            }
        }
        out
    }
}

/// How much of the walk's mass stays on the graph rather than restarting.
const WALK_DAMPING: f64 = 0.5;

/// Rounds of the walk. The graph is bounded at eight neighbours a node, so the
/// distribution settles well inside this.
const WALK_ROUNDS: usize = 20;

/// Follow each hit's stored links once, appending neighbours behind the hits.
///
/// The link graph is built by the write path and read by nothing that answers a
/// question, so whether it earns its place in retrieval has never been asked.
/// One hop behind the ranking is the cheapest way to ask: the ranking is
/// unchanged at the top and the neighbours can only fill places further down.
fn one_hop(ranked: &[String], atoms: &[Record], limit: usize) -> Vec<String> {
    let links: std::collections::HashMap<&str, Vec<&str>> = atoms
        .iter()
        .filter_map(|atom| {
            let id = atom.get("id").and_then(Value::as_str)?;
            let peers = atom
                .get("links")
                .and_then(Value::as_array)?
                .iter()
                .filter_map(Value::as_str)
                .collect();
            Some((id, peers))
        })
        .collect();
    let mut out: Vec<String> = ranked.to_vec();
    let mut seen: BTreeSet<String> = ranked.iter().cloned().collect();
    for id in ranked {
        for peer in links
            .get(id.as_str())
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            if out.len() >= limit {
                return out;
            }
            if seen.insert((*peer).to_string()) {
                out.push((*peer).to_string());
            }
        }
    }
    out
}

/// One table: a row per name, recall then hit at every cut-off.
fn table(names: &[&str], tallies: &[Tally]) {
    print!("{:<22}", "arm");
    for cut in CUTOFFS {
        print!("{:>10}", format!("R@{cut}"));
    }
    for cut in CUTOFFS {
        print!("{:>10}", format!("hit@{cut}"));
    }
    print!("{:>10}", format!("nDCG@{NDCG_CUT}"));
    println!();
    println!("{}", "-".repeat(32 + CUTOFFS.len() * 20));
    for (slot, name) in names.iter().enumerate() {
        let tally = &tallies[slot];
        let counted = tally.asked.max(1) as f64;
        print!("{name:<22}");
        for value in &tally.recall {
            print!("{:>10.3}", value / counted);
        }
        for value in &tally.hit {
            print!("{:>10.3}", *value as f64 / counted);
        }
        print!("{:>10.3}", tally.ndcg / counted);
        println!();
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "locomo10.json".to_string());
    let raw: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let mut corpus = conversations(&raw);
    anyhow::ensure!(!corpus.is_empty(), "no conversations in {path}");
    if let Some(cap) = conversation_cap() {
        corpus.truncate(cap);
        println!("scoring {} of the conversations, as asked", corpus.len());
    }

    let now = packset_core::clock::utcnow();
    let shipped = Panel::named("borda", "mmr", "off")?;
    let panels: Vec<Panel> = VOTERS
        .iter()
        .map(|name| Panel::named(name, "mmr", "off"))
        .collect::<Result<_, _>>()?;
    let mut totals: Vec<Tally> = ARMS.iter().map(|_| Tally::new()).collect();
    let mut by_session: Vec<Tally> = ARMS.iter().map(|_| Tally::new()).collect();
    let mut voted: Vec<Tally> = VOTERS.iter().map(|_| Tally::new()).collect();
    let mut voted_sessions: Vec<Tally> = VOTERS.iter().map(|_| Tally::new()).collect();
    let mut protocol: Vec<Tally> = PROTOCOLS.iter().map(|_| Tally::new()).collect();
    let mut room_voted: Vec<Tally> = VOTERS.iter().map(|_| Tally::new()).collect();
    // The pair the seat actually ships, swept the same way, because a default
    // argued from a pair the seat does not use is an argument about a
    // different retriever.
    let shipped_panels: Vec<Panel> = SHIPPED_VOTERS
        .iter()
        .map(|name| Panel::named(name, "mmr", "off"))
        .collect::<Result<_, _>>()?;
    let mut shipped_voted: Vec<Tally> = SHIPPED_VOTERS.iter().map(|_| Tally::new()).collect();
    let mut shipped_sessions: Vec<Tally> = SHIPPED_VOTERS.iter().map(|_| Tally::new()).collect();
    let mut ranking = Tally::new();
    let mut hopped = Tally::new();
    let mut walked = Tally::new();
    let mut turns = 0usize;
    let mut adversarial = 0usize;
    let mut linked = 0usize;
    let mut widest = 0usize;
    let encoder = packset_daemon::embed::binary();
    match &encoder {
        Some(path) => println!("encoder: {}", path.display()),
        None => println!("encoder: absent, so the dense arms will be empty"),
    }
    let encoder = encoder.is_some();
    let mut questions: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();
    let walking = walk_wanted();
    if walking {
        println!("restart walk: on, which is most of the time a run takes");
    }
    let late = encoder && late_wanted();
    if late {
        println!("late interaction: on, one vector per token");
    }
    let mut late_questions: std::collections::HashMap<String, Vec<Vec<f32>>> =
        std::collections::HashMap::new();
    // The same model's pooled output, so late interaction can be compared with
    // the model held fixed rather than against a different one.
    let mut m3_questions: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();
    // And the learned term weights, from that same pass.
    let mut sparse_questions: std::collections::HashMap<String, packset_daemon::embed::Sparse> =
        std::collections::HashMap::new();

    for (nth, conversation) in corpus.iter_mut().enumerate() {
        turns += conversation.atoms.len();
        // The write path is what builds the graph, so the benchmark runs it
        // rather than a copy of it: `apply_links` is what bounds the peer side
        // of an edge, and picking the neighbours without it produced a graph
        // with thirteen edges a turn against a cap of eight.
        let mut stored: Vec<Record> = Vec::with_capacity(conversation.atoms.len());
        for atom in &conversation.atoms {
            let mut fresh = atom.clone();
            let rewritten = packset_core::record::apply_links(
                &mut fresh,
                &stored,
                packset_core::record::LINK_THRESHOLD,
                &now,
            );
            for peer in rewritten {
                let Some(id) = peer.get("id").and_then(Value::as_str).map(str::to_string) else {
                    continue;
                };
                if let Some(slot) = stored
                    .iter_mut()
                    .find(|held| held.get("id").and_then(Value::as_str) == Some(id.as_str()))
                {
                    *slot = peer;
                }
            }
            stored.push(fresh);
        }
        conversation.atoms = stored;
        linked += conversation
            .atoms
            .iter()
            .map(|atom| {
                atom.get("links")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len)
            })
            .sum::<usize>();
        widest = widest.max(
            conversation
                .atoms
                .iter()
                .map(|atom| {
                    atom.get("links")
                        .and_then(Value::as_array)
                        .map_or(0, Vec::len)
                })
                .max()
                .unwrap_or(0),
        );
        // Built only when the walk runs: the adjacency costs a pass over every
        // atom's links, which a run that is not walking has no use for.
        let graph = walking.then(|| Graph::of(&conversation.atoms));
        let documents: Vec<Vec<String>> =
            conversation.atoms.iter().map(search::atom_tokens).collect();
        let index = Index::build(documents.iter().map(Vec::as_slice));
        // The same corpus with the session as the document rather than the turn.
        let session_corpus = session_documents(&conversation.atoms);
        let room_documents: Vec<Vec<String>> =
            session_corpus.iter().map(search::atom_tokens).collect();
        let room_index = Index::build(room_documents.iter().map(Vec::as_slice));

        // One encode of the corpus and one of the questions, through the kept
        // child the daemon uses. Absent encoder means the dense arms are empty
        // and the lexical ones still report, which is the seat's own fallback.
        if encoder {
            let model = model_name();
            let held = cache_dir();
            let atom_file = held
                .as_ref()
                .map(|dir| dir.join(format!("{model}-atoms-{nth}.vec")));
            let question_file = held
                .as_ref()
                .map(|dir| dir.join(format!("{model}-questions-{nth}.vec")));

            let cached = atom_file
                .as_deref()
                .and_then(|path| read_rows(path, conversation.atoms.len()));
            let vectors = cached.unwrap_or_else(|| {
                let fresh: Vec<Vec<f32>> = conversation
                    .atoms
                    .iter()
                    .map(|atom| {
                        let text = atom.get("text").and_then(Value::as_str).unwrap_or_default();
                        packset_daemon::embed::encode_document(text).unwrap_or_default()
                    })
                    .collect();
                if let Some(path) = atom_file.as_deref() {
                    write_rows(path, &fresh);
                }
                fresh
            });
            for (atom, vector) in conversation.atoms.iter_mut().zip(vectors) {
                if vector.is_empty() {
                    continue;
                }
                atom.insert(
                    "embedding".into(),
                    Value::Array(vector.into_iter().map(|f| json!(f)).collect()),
                );
            }

            let cached = question_file
                .as_deref()
                .and_then(|path| read_rows(path, conversation.questions.len()));
            let asked = cached.unwrap_or_else(|| {
                let fresh: Vec<Vec<f32>> = conversation
                    .questions
                    .iter()
                    .map(|question| {
                        packset_daemon::embed::encode_query(&question.text).unwrap_or_default()
                    })
                    .collect();
                if let Some(path) = question_file.as_deref() {
                    write_rows(path, &fresh);
                }
                fresh
            });
            for (question, vector) in conversation.questions.iter().zip(asked) {
                if vector.is_empty() {
                    continue;
                }
                questions.insert(question.text.clone(), vector);
            }
        }

        // Per-token vectors live beside the atoms rather than inside them: a
        // thousand floats per token would not fit an atom's JSON without
        // making every other read pay for it.
        let mut late_atoms: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut m3_atoms: Vec<Vec<f32>> = Vec::new();
        let mut sparse_atoms: Vec<packset_daemon::embed::Sparse> = Vec::new();
        if late {
            for atom in &conversation.atoms {
                let text = atom.get("text").and_then(Value::as_str).unwrap_or_default();
                let (tokens, pooled, weights) =
                    packset_daemon::embed::encode_late(text).unwrap_or_default();
                late_atoms.push(tokens);
                m3_atoms.push(pooled);
                sparse_atoms.push(weights);
            }
            for question in &conversation.questions {
                if late_questions.contains_key(question.text.as_str()) {
                    continue;
                }
                if let Some((tokens, pooled, weights)) =
                    packset_daemon::embed::encode_late(&question.text)
                {
                    late_questions.insert(question.text.clone(), tokens);
                    m3_questions.insert(question.text.clone(), pooled);
                    sparse_questions.insert(question.text.clone(), weights);
                }
            }
        }

        for question in &conversation.questions {
            if question.category == ADVERSARIAL || question.evidence.is_empty() {
                adversarial += 1;
                continue;
            }
            let ask = Ask {
                user: "",
                memory: "",
                atoms: &conversation.atoms,
                query: &question.text,
                // Ranked deeper than the deepest cut-off, so a cut-off is a cut
                // rather than the whole list.
                limit: CUTOFFS[CUTOFFS.len() - 1],
                set: None,
                now: &now,
            };
            let lexical = search::search_linear(&ask);
            let terms = search::search_bm25(&ask, &index);
            let fed = search::search_bm25_expanded(&ask, &index, &documents);
            let meaning = questions
                .get(question.text.as_str())
                .map(|vector| search::search_dense(&ask, vector))
                .unwrap_or_default();
            let interaction = late_questions
                .get(question.text.as_str())
                .map(|tokens| rank_late(&ask, tokens, &late_atoms))
                .unwrap_or_default();
            let m3_pooled = m3_questions
                .get(question.text.as_str())
                .map(|vector| rank_pooled(&ask, vector, &m3_atoms))
                .unwrap_or_default();
            let m3_sparse = sparse_questions
                .get(question.text.as_str())
                .map(|weights| rank_sparse(&ask, weights, &sparse_atoms))
                .unwrap_or_default();

            // Does the stored graph earn a place in an answer? Half the places
            // are the ranking's, and the rest go either to the ranking
            // continuing or to the neighbours of what it already found. Same
            // budget, same question, one difference.

            let deep = hit_ids(&search::merge_ballots(
                &[lexical.clone(), terms.clone()],
                HOP_CUT,
                &shipped,
                &now,
            ));
            ranking.add(&deep, &question.evidence);
            let shallow: Vec<String> = deep.iter().take(HOP_CUT / 2).cloned().collect();
            hopped.add(
                &one_hop(&shallow, &conversation.atoms, HOP_CUT),
                &question.evidence,
            );
            if let Some(graph) = graph.as_ref() {
                walked.add(&graph.expand(&shallow, HOP_CUT), &question.evidence);
            }

            for (slot, arm) in ARMS.iter().enumerate() {
                let ranked = match *arm {
                    "lexical" => hit_ids(&lexical),
                    "bm25" => hit_ids(&terms),
                    "bm25 rm3" => hit_ids(&fed),
                    "dense" => hit_ids(&meaning),
                    "late" => hit_ids(&interaction),
                    "m3 pooled" => hit_ids(&m3_pooled),
                    "m3 sparse" => hit_ids(&m3_sparse),
                    other => {
                        let ballots = match other {
                            "bm25+late" => vec![terms.clone(), interaction.clone()],
                            "bm25+m3 pooled" => vec![terms.clone(), m3_pooled.clone()],
                            "bm25+m3 sparse" => vec![terms.clone(), m3_sparse.clone()],
                            "m3 sparse+late" => vec![m3_sparse.clone(), interaction.clone()],
                            "bm25+dense" => vec![terms.clone(), meaning.clone()],
                            "bm25 rm3+dense" => vec![fed.clone(), meaning.clone()],
                            "lexical+bm25+dense" => {
                                vec![lexical.clone(), terms.clone(), meaning.clone()]
                            }
                            _ => vec![lexical.clone(), terms.clone()],
                        };
                        hit_ids(&search::merge_ballots(&ballots, ask.limit, &shipped, &now))
                    }
                };
                totals[slot].add(&ranked, &question.evidence);
                let rooms: BTreeSet<String> = question
                    .evidence
                    .iter()
                    .map(|id| session_of(id).to_string())
                    .collect();
                by_session[slot].add(&sessions_of(&ranked), &rooms);
            }

            // The strongest pair this run has, fused every way the panel
            // knows. Same ballots, same questions, one difference.
            let pair = if interaction.is_empty() {
                if meaning.is_empty() {
                    vec![lexical.clone(), terms.clone()]
                } else {
                    vec![terms.clone(), meaning.clone()]
                }
            } else {
                vec![terms.clone(), interaction.clone()]
            };
            let rooms: BTreeSet<String> = question
                .evidence
                .iter()
                .map(|id| session_of(id).to_string())
                .collect();
            for (slot, panel) in panels.iter().enumerate() {
                let ranked = hit_ids(&search::merge_ballots(&pair, ask.limit, panel, &now));
                voted[slot].add(&ranked, &question.evidence);
                voted_sessions[slot].add(&sessions_of(&ranked), &rooms);
            }
            let shipped_pair = if meaning.is_empty() {
                vec![lexical.clone(), terms.clone()]
            } else {
                vec![lexical.clone(), terms.clone(), meaning.clone()]
            };
            for (slot, panel) in shipped_panels.iter().enumerate() {
                let ranked = hit_ids(&search::merge_ballots(
                    &shipped_pair,
                    ask.limit,
                    panel,
                    &now,
                ));
                shipped_voted[slot].add(&ranked, &question.evidence);
                shipped_sessions[slot].add(&sessions_of(&ranked), &rooms);
            }

            // The same question against a corpus of sessions, so what varies
            // between these arms is what a document is.
            let asking = Ask {
                atoms: &session_corpus,
                ..ask
            };
            let room_terms = search::search_bm25(&asking, &room_index);
            let room_fed = search::search_bm25_expanded(&asking, &room_index, &room_documents);
            let by_turn = collapse(&terms);
            let by_meaning = collapse(&meaning);
            let by_late = collapse(&interaction);
            let by_sparse = collapse(&m3_sparse);
            for (slot, arm) in PROTOCOLS.iter().enumerate() {
                let ranked = match *arm {
                    "turn bm25" => hit_ids(&by_turn),
                    "session bm25" => hit_ids(&room_terms),
                    "session bm25 rm3" => hit_ids(&room_fed),
                    "turn dense" => hit_ids(&by_meaning),
                    "session bm25 + turn late" => hit_ids(&search::merge_ballots(
                        &[room_terms.clone(), by_late.clone()],
                        ask.limit,
                        &shipped,
                        &now,
                    )),
                    "session bm25 + turn m3 sparse" => hit_ids(&search::merge_ballots(
                        &[room_terms.clone(), by_sparse.clone()],
                        ask.limit,
                        &shipped,
                        &now,
                    )),
                    _ => hit_ids(&search::merge_ballots(
                        &[room_terms.clone(), by_meaning.clone()],
                        ask.limit,
                        &shipped,
                        &now,
                    )),
                };
                protocol[slot].add(&ranked, &rooms);
            }
            // And the fusion question again, on the pair this table says is
            // strongest, because the published gain is credited to the fusion
            // rather than to either retriever.
            let room_pair = if by_late.is_empty() {
                vec![room_terms.clone(), by_meaning.clone()]
            } else {
                vec![room_terms.clone(), by_late.clone()]
            };
            for (slot, panel) in panels.iter().enumerate() {
                let ranked = hit_ids(&search::merge_ballots(&room_pair, ask.limit, panel, &now));
                room_voted[slot].add(&ranked, &rooms);
            }
        }
    }

    let asked = totals[0].asked;
    println!(
        "{} conversations, {turns} turns, {asked} answerable questions \
         ({adversarial} adversarial or unlabelled, excluded)",
        corpus.len()
    );
    println!(
        "link graph: {linked} edges, {:.1} per turn, widest {widest}",
        linked as f64 / turns.max(1) as f64
    );
    println!();
    table(ARMS, &totals);

    println!();
    println!("the same rankings read as sessions, which is the unit the");
    println!("retrieval papers on this benchmark score:");
    println!();
    table(ARMS, &by_session);

    let swept = if late {
        "bm25+late"
    } else if encoder {
        "bm25+dense"
    } else {
        "lexical+bm25"
    };
    println!();
    println!("{swept}, fused every way the panel knows, by turn:");
    println!();
    table(VOTERS, &voted);
    println!();
    println!("the same, by session:");
    println!();
    table(VOTERS, &voted_sessions);

    println!();
    println!(
        "the ballots the seat ships ({}), fused every way, by turn:",
        if encoder {
            "lexical + bm25 + dense"
        } else {
            "lexical + bm25"
        }
    );
    println!();
    table(SHIPPED_VOTERS, &shipped_voted);
    println!();
    println!("the same, by session:");
    println!();
    table(SHIPPED_VOTERS, &shipped_sessions);

    println!();
    println!("a session as the document, against a session read off a turn");
    println!("ranking, which are two protocols behind one word:");
    println!();
    table(PROTOCOLS, &protocol);
    println!();
    println!(
        "{}, fused every way the panel knows:",
        if late {
            "session bm25 + turn late"
        } else {
            "session bm25 + turn dense"
        }
    );
    println!();
    table(VOTERS, &room_voted);

    let slot = CUTOFFS.iter().position(|cut| *cut == HOP_CUT).expect("cut");
    let counted = ranking.asked.max(1) as f64;
    println!();
    println!(
        "the last {} places of {HOP_CUT}, given to the ranking or to the graph:",
        HOP_CUT / 2
    );
    println!(
        "  ranking continues      R@{HOP_CUT} {:.3}   hit@{HOP_CUT} {:.3}",
        ranking.recall[slot] / counted,
        ranking.hit[slot] as f64 / counted
    );
    println!(
        "  neighbours of the top  R@{HOP_CUT} {:.3}   hit@{HOP_CUT} {:.3}",
        hopped.recall[slot] / counted,
        hopped.hit[slot] as f64 / counted
    );
    if walking {
        println!(
            "  a restart walk from it R@{HOP_CUT} {:.3}   hit@{HOP_CUT} {:.3}",
            walked.recall[slot] / counted,
            walked.hit[slot] as f64 / counted
        );
    }

    let mut merged = Tally::new();
    merged.merge(&totals[0]);
    anyhow::ensure!(
        merged.asked == asked,
        "the arms answered different questions"
    );
    Ok(())
}
