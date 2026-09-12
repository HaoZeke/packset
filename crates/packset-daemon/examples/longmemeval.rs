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

use packset_core::bm25::Index;
use packset_core::search::{atom_tokens, Record};
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

/// Documents for one protocol: each with the session it belongs to.
fn documents(question: &Question, protocol: &str) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    for (sid, turns) in &question.sessions {
        match protocol {
            "turns" => {
                for turn in turns {
                    out.push((sid.clone(), tokens(turn)));
                }
            }
            "windows" => {
                if turns.len() <= WINDOW {
                    out.push((sid.clone(), tokens(&turns.join("\n"))));
                } else {
                    let mut start = 0;
                    while start < turns.len() {
                        let end = (start + WINDOW).min(turns.len());
                        out.push((sid.clone(), tokens(&turns[start..end].join("\n"))));
                        if end == turns.len() {
                            break;
                        }
                        start += STRIDE;
                    }
                }
            }
            _ => out.push((sid.clone(), tokens(&turns.join("\n")))),
        }
    }
    out
}

/// Sessions in the order their best document ranks.
fn ranked_sessions(question: &Question, protocol: &str) -> Vec<String> {
    let docs = documents(question, protocol);
    let index = Index::build(docs.iter().map(|(_, t)| t.as_slice()));
    let mut scored = index.score(&tokens(&question.text));
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for (ordinal, _) in scored {
        let sid = &docs[ordinal].0;
        if seen.insert(sid.clone()) {
            out.push(sid.clone());
        }
    }
    out
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
    let started = std::time::Instant::now();
    let mut overall: Vec<Tally> = PROTOCOLS.iter().map(|_| Tally::new()).collect();
    let mut by_kind: BTreeMap<String, Vec<Tally>> = BTreeMap::new();
    for question in &asked {
        for (slot, protocol) in PROTOCOLS.iter().enumerate() {
            let ranked = ranked_sessions(question, protocol);
            overall[slot].add(&ranked, &question.answers);
            by_kind
                .entry(question.kind.clone())
                .or_insert_with(|| PROTOCOLS.iter().map(|_| Tally::new()).collect())[slot]
                .add(&ranked, &question.answers);
        }
    }
    let header = {
        let mut h = String::from("| arm | asked |");
        for cut in CUTOFFS {
            h.push_str(&format!(" hit@{cut} | recall@{cut} |"));
        }
        h
    };
    println!("\nLongMemEval_S, BM25+ over three document protocols, session granularity\n");
    println!("{header}");
    println!("|---|---|{}", "---|---|".repeat(CUTOFFS.len()));
    for (slot, protocol) in PROTOCOLS.iter().enumerate() {
        println!("{}", overall[slot].row(protocol));
    }
    for (kind, tallies) in &by_kind {
        println!("\n{kind}\n\n{header}");
        println!("|---|---|{}", "---|---|".repeat(CUTOFFS.len()));
        for (slot, protocol) in PROTOCOLS.iter().enumerate() {
            println!("{}", tallies[slot].row(protocol));
        }
    }
    println!(
        "\n{} questions in {:.1}s",
        asked.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
