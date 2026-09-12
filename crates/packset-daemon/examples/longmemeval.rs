//! Session retrieval on LongMemEval_S (doi:10.48550/arXiv.2410.10813): 500
//! questions, each over about fifty chat sessions, with the sessions that
//! answer it labelled. Turns are loaded as atoms, so this measures the lexical
//! scorer under three document protocols and nothing else: no model reads
//! the retrieved context, and nothing is extracted on write. Abstention
//! questions (`_abs`) have no answer session and are excluded.
//!
//! ```console
//! $ curl -sL -o longmemeval_s.json https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s
//! $ cargo run --release -p packset-daemon --example longmemeval -- longmemeval_s.json
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};

use packset_core::bm25::Index;
use packset_core::panel::Panel;
use packset_core::search::{atom_tokens, cosine, merge_ballots, Record};
use serde_json::{json, Value};

const CUTOFFS: &[usize] = &[1, 5, 10];

/// Passage windows: six turns, stride three, as the LoCoMo protocol table settled.
const WINDOW: usize = 6;
const STRIDE: usize = 3;

struct Question {
    kind: String,
    text: String,
    /// Session id, then its turns in order.
    sessions: Vec<(String, Vec<String>)>,
    answers: BTreeSet<String>,
}

fn questions(raw: &Value) -> Vec<Question> {
    let Some(items) = raw.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|q| {
            let id = q["question_id"].as_str()?;
            if id.ends_with("_abs") {
                return None;
            }
            let ids = q["haystack_session_ids"].as_array()?;
            let sessions = q["haystack_sessions"].as_array()?;
            let sessions = ids
                .iter()
                .zip(sessions)
                .filter_map(|(sid, turns)| {
                    let turns: Vec<String> = turns
                        .as_array()?
                        .iter()
                        .filter_map(|t| {
                            let role = t["role"].as_str().unwrap_or("");
                            let content = t["content"].as_str()?;
                            Some(format!("{role}: {content}"))
                        })
                        .collect();
                    Some((sid.as_str()?.to_string(), turns))
                })
                .collect();
            let answers = q["answer_session_ids"]
                .as_array()?
                .iter()
                .filter_map(|a| a.as_str().map(str::to_string))
                .collect();
            Some(Question {
                kind: q["question_type"].as_str().unwrap_or("?").to_string(),
                text: q["question"].as_str()?.to_string(),
                sessions,
                answers,
            })
        })
        .collect()
}

fn tokens(text: &str) -> Vec<String> {
    let record: Record = json!({"text": text})
        .as_object()
        .cloned()
        .unwrap_or_default();
    atom_tokens(&record)
}

/// One document: the session it belongs to, its text, its tokens.
struct Document {
    session: String,
    text: String,
    tokens: Vec<String>,
}

/// Documents for one protocol.
fn documents(question: &Question, protocol: &str) -> Vec<Document> {
    let doc = |sid: &str, text: String| Document {
        session: sid.to_string(),
        tokens: tokens(&text),
        text,
    };
    let mut out = Vec::new();
    for (sid, turns) in &question.sessions {
        match protocol {
            "turns" => {
                for turn in turns {
                    out.push(doc(sid, turn.clone()));
                }
            }
            "windows" => {
                if turns.len() <= WINDOW {
                    out.push(doc(sid, turns.join("\n")));
                } else {
                    let mut start = 0;
                    while start < turns.len() {
                        let end = (start + WINDOW).min(turns.len());
                        out.push(doc(sid, turns[start..end].join("\n")));
                        if end == turns.len() {
                            break;
                        }
                        start += STRIDE;
                    }
                }
            }
            _ => out.push(doc(sid, turns.join("\n"))),
        }
    }
    out
}

/// BM25+ over the documents: ordinal and score, best first.
fn lexical(question: &Question, docs: &[Document]) -> Vec<(usize, f64)> {
    let index = Index::build(docs.iter().map(|d| d.tokens.as_slice()));
    let mut scored = index.score(&tokens(&question.text));
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored
}

/// Cosine over document vectors: ordinal and score, best first.
fn dense(query: &[f32], vectors: &[Vec<f32>]) -> Vec<(usize, f64)> {
    let mut scored: Vec<(usize, f64)> = vectors
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_empty())
        .map(|(i, v)| (i, cosine(query, v)))
        .filter(|(_, s)| *s > 0.0)
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored
}

/// Two rankings fused by the shipped panel (CombMNZ), as ordinals.
fn fused(lexical: &[(usize, f64)], dense: &[(usize, f64)], limit: usize) -> Vec<usize> {
    let ballot = |ranked: &[(usize, f64)]| -> Vec<Value> {
        ranked
            .iter()
            .take(limit)
            .map(|(i, s)| json!({"field": "atom", "id": i.to_string(), "text": "", "score": s}))
            .collect()
    };
    let panel = Panel::named("combmnz", "none", "off").expect("a shipped panel");
    let now = packset_core::clock::utcnow();
    merge_ballots(&[ballot(lexical), ballot(dense)], limit, &panel, &now)
        .iter()
        .filter_map(|hit| hit["id"].as_str()?.parse().ok())
        .collect()
}

/// Sessions in the order their best document ranks.
fn collapse(order: impl Iterator<Item = usize>, docs: &[Document]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for ordinal in order {
        let sid = &docs[ordinal].session;
        if seen.insert(sid.clone()) {
            out.push(sid.clone());
        }
    }
    out
}

/// One protocol's documents and their lexical ranking, kept for the dense arms.
type Scored = (Vec<Document>, Vec<(usize, f64)>);

/// How many documents each ballot hands the fuse.
const FUSE_DEPTH: usize = 50;

fn cache_dir() -> Option<std::path::PathBuf> {
    let dir = std::path::PathBuf::from(std::env::var_os("PACKSET_LME_CACHE")?);
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

fn write_rows(path: &std::path::Path, rows: &[Vec<f32>]) {
    let Ok(mut file) = std::fs::File::create(path) else {
        return;
    };
    let _ = file.write_all(&(rows.len() as u64).to_le_bytes());
    for row in rows {
        let _ = file.write_all(&(row.len() as u64).to_le_bytes());
        for v in row {
            let _ = file.write_all(&v.to_le_bytes());
        }
    }
}

fn read_rows(path: &std::path::Path, expected: usize) -> Option<Vec<Vec<f32>>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .read_to_end(&mut bytes)
        .ok()?;
    let mut at = 0usize;
    let count = u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?) as usize;
    at += 8;
    if count != expected {
        return None;
    }
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count {
        let len = u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?) as usize;
        at += 8;
        let mut row = Vec::with_capacity(len);
        for _ in 0..len {
            row.push(f32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?));
            at += 4;
        }
        rows.push(row);
    }
    Some(rows)
}

/// Document vectors for one question and protocol, from the cache when the
/// count matches, else encoded and cached.
fn vectors(nth: usize, protocol: &str, docs: &[Document]) -> Vec<Vec<f32>> {
    let model = std::env::var("PACKSET_EMBED_MODEL").unwrap_or_else(|_| "default".into());
    let file = cache_dir().map(|d| d.join(format!("{model}-{protocol}-q{nth}.bin")));
    if let Some(rows) = file.as_deref().and_then(|p| read_rows(p, docs.len())) {
        return rows;
    }
    let fresh: Vec<Vec<f32>> = docs
        .iter()
        .map(|d| packset_daemon::embed::encode_document(&d.text).unwrap_or_default())
        .collect();
    if let Some(path) = file.as_deref() {
        write_rows(path, &fresh);
    }
    fresh
}

#[derive(Default, Clone)]
struct Tally {
    asked: usize,
    /// Any answer session inside the cut-off.
    hit: Vec<usize>,
    /// Fraction of answer sessions inside the cut-off, summed.
    recall: Vec<f64>,
}

impl Tally {
    fn new() -> Self {
        Self {
            asked: 0,
            hit: vec![0; CUTOFFS.len()],
            recall: vec![0.0; CUTOFFS.len()],
        }
    }

    fn add(&mut self, ranked: &[String], answers: &BTreeSet<String>) {
        self.asked += 1;
        for (slot, &cut) in CUTOFFS.iter().enumerate() {
            let top: BTreeSet<&String> = ranked.iter().take(cut).collect();
            let found = answers.iter().filter(|a| top.contains(a)).count();
            if found > 0 {
                self.hit[slot] += 1;
            }
            self.recall[slot] += found as f64 / answers.len().max(1) as f64;
        }
    }

    fn row(&self, name: &str) -> String {
        let mut line = format!("| {name} | {} |", self.asked);
        for slot in 0..CUTOFFS.len() {
            line.push_str(&format!(
                " {:.3} | {:.3} |",
                self.hit[slot] as f64 / self.asked.max(1) as f64,
                self.recall[slot] / self.asked.max(1) as f64
            ));
        }
        line
    }
}

const PROTOCOLS: &[&str] = &["turns", "windows", "sessions"];

/// The protocol the dense and fused arms run on. Turns and windows are too
/// many to encode on a CPU: a hundred questions of windows did not finish in
/// four hours, fifty session documents a question do.
const DENSE_PROTOCOLS: &[&str] = &["sessions"];

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "longmemeval_s.json".to_string());
    let raw: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let mut asked = questions(&raw);
    anyhow::ensure!(!asked.is_empty(), "no questions in {path}");
    if let Some(cap) = std::env::var("PACKSET_LME_QUESTIONS")
        .ok()
        .and_then(|c| c.parse().ok())
    {
        asked.truncate(cap);
        println!("scoring {} questions, as asked", asked.len());
    }
    let encoder = packset_daemon::embed::binary().is_some();
    println!(
        "encoder: {}",
        if encoder {
            "present, dense and fused arms run on session documents"
        } else {
            "absent, lexical arms only"
        }
    );
    let mut arms: Vec<String> = PROTOCOLS.iter().map(|p| p.to_string()).collect();
    if encoder {
        for p in DENSE_PROTOCOLS {
            arms.push(format!("{p} dense"));
            arms.push(format!("{p} fused"));
        }
    }
    let started = std::time::Instant::now();
    let mut overall: Vec<Tally> = arms.iter().map(|_| Tally::new()).collect();
    let mut by_kind: BTreeMap<String, Vec<Tally>> = BTreeMap::new();
    for (nth, question) in asked.iter().enumerate() {
        let mut slot = 0usize;
        let mut record = |ranked: &[String], slot: usize| {
            overall[slot].add(ranked, &question.answers);
            by_kind
                .entry(question.kind.clone())
                .or_insert_with(|| arms.iter().map(|_| Tally::new()).collect())[slot]
                .add(ranked, &question.answers);
        };
        let mut kept: BTreeMap<&str, Scored> = BTreeMap::new();
        for protocol in PROTOCOLS {
            let docs = documents(question, protocol);
            let lex = lexical(question, &docs);
            record(&collapse(lex.iter().map(|(i, _)| *i), &docs), slot);
            slot += 1;
            kept.insert(protocol, (docs, lex));
        }
        if encoder {
            let query = packset_daemon::embed::encode_query(&question.text).unwrap_or_default();
            for protocol in DENSE_PROTOCOLS {
                let (docs, lex) = &kept[protocol];
                let vecs = vectors(nth, protocol, docs);
                let den = dense(&query, &vecs);
                record(&collapse(den.iter().map(|(i, _)| *i), docs), slot);
                slot += 1;
                record(
                    &collapse(fused(lex, &den, FUSE_DEPTH).into_iter(), docs),
                    slot,
                );
                slot += 1;
            }
        }
        if (nth + 1) % 50 == 0 {
            eprintln!(
                "{} questions, {:.0}s",
                nth + 1,
                started.elapsed().as_secs_f64()
            );
        }
    }
    let header = {
        let mut h = String::from("| arm | asked |");
        for cut in CUTOFFS {
            h.push_str(&format!(" hit@{cut} | recall@{cut} |"));
        }
        h
    };
    println!("\nLongMemEval_S, session granularity\n");
    println!("{header}");
    println!("|---|---|{}", "---|---|".repeat(CUTOFFS.len()));
    for (slot, arm) in arms.iter().enumerate() {
        println!("{}", overall[slot].row(arm));
    }
    for (kind, tallies) in &by_kind {
        println!("\n{kind}\n\n{header}");
        println!("|---|---|{}", "---|---|".repeat(CUTOFFS.len()));
        for (slot, arm) in arms.iter().enumerate() {
            println!("{}", tallies[slot].row(arm));
        }
    }
    println!(
        "\n{} questions in {:.1}s",
        asked.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
