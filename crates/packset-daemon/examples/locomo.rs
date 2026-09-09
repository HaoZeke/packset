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
    /// Questions counted.
    asked: usize,
}

impl Tally {
    fn new() -> Self {
        Self {
            recall: vec![0.0; CUTOFFS.len()],
            hit: vec![0; CUTOFFS.len()],
            asked: 0,
        }
    }

    fn add(&mut self, ranked: &[String], evidence: &BTreeSet<String>) {
        self.asked += 1;
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

/// The arms compared, each a way of turning one question into a ranking.
const ARMS: &[&str] = &[
    "lexical",
    "bm25",
    "dense",
    "lexical+bm25 (shipped)",
    "lexical+bm25+dense",
    "bm25+dense",
];

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

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "locomo10.json".to_string());
    let raw: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let mut corpus = conversations(&raw);
    anyhow::ensure!(!corpus.is_empty(), "no conversations in {path}");

    let now = packset_core::clock::utcnow();
    let shipped = Panel::named("borda", "mmr", "off")?;
    let mut totals: Vec<Tally> = ARMS.iter().map(|_| Tally::new()).collect();
    let mut by_session: Vec<Tally> = ARMS.iter().map(|_| Tally::new()).collect();
    let mut ranking = Tally::new();
    let mut hopped = Tally::new();
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

    for conversation in &mut corpus {
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
        let documents: Vec<Vec<String>> =
            conversation.atoms.iter().map(search::atom_tokens).collect();
        let index = Index::build(documents.iter().map(Vec::as_slice));

        // One encode of the corpus and one of the questions, through the kept
        // child the daemon uses. Absent encoder means the dense arms are empty
        // and the lexical ones still report, which is the seat's own fallback.
        if encoder {
            for atom in &mut conversation.atoms {
                let text = atom
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if let Some(vector) = packset_daemon::embed::encode_document(&text) {
                    atom.insert(
                        "embedding".into(),
                        Value::Array(vector.into_iter().map(|f| json!(f)).collect()),
                    );
                }
            }
            for question in &conversation.questions {
                if questions.contains_key(question.text.as_str()) {
                    continue;
                }
                if let Some(vector) = packset_daemon::embed::encode_query(&question.text) {
                    questions.insert(question.text.clone(), vector);
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
            let meaning = questions
                .get(question.text.as_str())
                .map(|vector| search::search_dense(&ask, vector))
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

            for (slot, arm) in ARMS.iter().enumerate() {
                let ranked = match *arm {
                    "lexical" => hit_ids(&lexical),
                    "bm25" => hit_ids(&terms),
                    "dense" => hit_ids(&meaning),
                    other => {
                        let ballots = match other {
                            "bm25+dense" => vec![terms.clone(), meaning.clone()],
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
    print!("{:<22}", "arm");
    for cut in CUTOFFS {
        print!("{:>10}", format!("R@{cut}"));
    }
    for cut in CUTOFFS {
        print!("{:>10}", format!("hit@{cut}"));
    }
    println!();
    println!("{}", "-".repeat(22 + CUTOFFS.len() * 20));
    for (slot, arm) in ARMS.iter().enumerate() {
        let tally = &totals[slot];
        let asked = tally.asked.max(1) as f64;
        print!("{arm:<22}");
        for value in &tally.recall {
            print!("{:>10.3}", value / asked);
        }
        for value in &tally.hit {
            print!("{:>10.3}", *value as f64 / asked);
        }
        println!();
    }

    println!();
    println!("the same rankings read as sessions, which is the unit the");
    println!("retrieval papers on this benchmark score:");
    println!();
    print!("{:<22}", "arm");
    for cut in CUTOFFS {
        print!("{:>10}", format!("R@{cut}"));
    }
    for cut in CUTOFFS {
        print!("{:>10}", format!("hit@{cut}"));
    }
    println!();
    println!("{}", "-".repeat(22 + CUTOFFS.len() * 20));
    for (slot, arm) in ARMS.iter().enumerate() {
        let tally = &by_session[slot];
        let counted = tally.asked.max(1) as f64;
        print!("{arm:<22}");
        for value in &tally.recall {
            print!("{:>10.3}", value / counted);
        }
        for value in &tally.hit {
            print!("{:>10.3}", *value as f64 / counted);
        }
        println!();
    }

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

    let mut merged = Tally::new();
    merged.merge(&totals[0]);
    anyhow::ensure!(
        merged.asked == asked,
        "the arms answered different questions"
    );
    Ok(())
}
