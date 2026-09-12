//! A synthetic longitudinal corpus for the decay slot: does a claim the seat
//! kept recalling outrank paraphrases it wrote later and never used?
//!
//! Every topic has four claims that share its words. One, written early, is
//! recalled on its review clock across the run; the other three are written
//! late and never reviewed. At the end the topic is asked as a cue. Recency
//! favours the late paraphrases, the review clock favours the kept claim, and
//! a lexical scorer cannot tell them apart. LoCoMo and LongMemEval carry no
//! review history, so this is the corpus that can measure the slot at all.
//!
//! ```console
//! $ cargo run --release -p packset-daemon --example forgetting
//! ```

use packset_core::bm25::Index;
use packset_core::panel::Panel;
use packset_core::record::{schedule_review, Grade};
use packset_core::search::{atom_tokens, merge_ballots, Record};
use serde_json::{json, Value};

const TOPICS: usize = 300;
const PARAPHRASES: usize = 3;
const DAYS: i64 = 180;
const BASE: &str = "2026-01-01T00:00:00.000Z";

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn day(t: i64) -> String {
    packset_core::clock::shift(BASE, t * 86_400).expect("a stamp")
}

fn claim(topic: usize, variant: usize, written: i64) -> Record {
    let text = format!(
        "On topic{topic} the settled answer is variant{variant}. It was checked against detail{variant}."
    );
    let mut atom: Record = json!({
        "id": format!("t{topic}-v{variant}"),
        "field": "atom",
        "kind": "lesson",
        "text": text,
        "ts": day(written),
    })
    .as_object()
    .cloned()
    .expect("an object");
    schedule_review(&mut atom, &day(written), Grade::Initial, None);
    atom
}

/// Recall the kept claim whenever it falls due, up to `until`.
fn keep_recalling(atom: &mut Record, until: i64) {
    for _ in 0..64 {
        let due = atom["due_at"].as_str().unwrap_or("").to_string();
        let Some(due_ms) = packset_core::clock::parse_millis(&due) else {
            break;
        };
        let base_ms = packset_core::clock::parse_millis(BASE).unwrap_or(0);
        let due_day = (due_ms - base_ms) / 86_400_000;
        if due_day > until {
            break;
        }
        schedule_review(atom, &due, Grade::Recalled, None);
    }
}

fn hits_for(cue: &[String], index: &Index, atoms: &[Record]) -> Vec<Value> {
    let mut scored = index.score(cue);
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored
        .into_iter()
        .take(20)
        .map(|(ordinal, score)| {
            let mut hit = atoms[ordinal].clone();
            hit.insert("score".into(), json!(score));
            Value::Object(hit)
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut atoms: Vec<Record> = Vec::new();
    for topic in 0..TOPICS {
        let early = rng.below(30) as i64;
        let mut kept = claim(topic, 0, early);
        keep_recalling(&mut kept, DAYS);
        atoms.push(kept);
        for variant in 1..=PARAPHRASES {
            let late = 150 + rng.below(30) as i64;
            atoms.push(claim(topic, variant, late));
        }
    }
    let documents: Vec<Vec<String>> = atoms.iter().map(atom_tokens).collect();
    let index = Index::build(documents.iter().map(Vec::as_slice));
    let now = day(DAYS);
    let panels = [
        (
            "off (lexical only)",
            Panel::named("combmnz", "none", "off")?,
        ),
        (
            "on (14-day half-life on age)",
            Panel::named("combmnz", "none", "on")?,
        ),
        (
            "fsrs (retrievability)",
            Panel::named("combmnz", "none", "fsrs")?,
        ),
    ];
    println!(
        "{TOPICS} topics, one kept claim written in the first 30 days and recalled on its clock, {PARAPHRASES} paraphrases written after day 150 and never reviewed; asked on day {DAYS}\n"
    );
    println!("| decay slot | kept claim first | mean rank of the kept claim |");
    println!("|---|---|---|");
    for (name, panel) in &panels {
        let mut first = 0usize;
        let mut rank_sum = 0usize;
        for topic in 0..TOPICS {
            let cue = atom_tokens(
                json!({"text": format!("topic{topic} settled answer")})
                    .as_object()
                    .expect("an object"),
            );
            let ranked = merge_ballots(&[hits_for(&cue, &index, &atoms)], 10, panel, &now);
            let kept = format!("t{topic}-v0");
            let rank = ranked
                .iter()
                .position(|h| h["id"].as_str() == Some(kept.as_str()))
                .map_or(10, |r| r + 1);
            if rank == 1 {
                first += 1;
            }
            rank_sum += rank;
        }
        println!(
            "| {name} | {:.3} | {:.2} |",
            first as f64 / TOPICS as f64,
            rank_sum as f64 / TOPICS as f64
        );
    }
    Ok(())
}
