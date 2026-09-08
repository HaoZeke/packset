//! Lexical scoring over a pack.
//!
//! Two scorers answer the same question. This one is a prefix-and-one-edit
//! scan, which is right for the tens of atoms a seat actually holds; the milli
//! projection takes over when the index is there. Both feed the same merge, so
//! which one ran is a fact about the machine rather than about the ranking.

use serde_json::{json, Map, Value};

use crate::{clock, record};

/// A query token shorter than this matches exactly or not at all.
pub const SHORT_EXACT: usize = 4;
/// Days over which a hit's recency weight halves.
pub const RECENCY_HALF_LIFE_DAYS: f64 = 14.0;

/// Words carrying no signal, dropped from a query and from the text.
pub const STOP: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could", "do",
    "does", "for", "from", "if", "in", "is", "it", "its", "me", "my", "no", "not", "of", "on",
    "or", "our", "please", "should", "than", "that", "the", "then", "these", "this", "those", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "will", "with", "would", "yes",
    "you", "your",
];

/// One atom.
pub type Record = Map<String, Value>;

/// Lowercase tokens with the stopwords dropped.
#[must_use]
pub fn tokens(text: &str) -> Vec<String> {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphanumeric() {
            let start = i;
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-')
            {
                i += 1;
            }
            let token = &lower[start..i];
            if !STOP.contains(&token) {
                out.push(token.to_string());
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Paragraphs, or lines when the text is one block.
///
/// A card written as a list of one-line claims would otherwise score as a
/// single paragraph and return the whole file as one hit.
#[must_use]
pub fn paragraphs(text: &str) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                let joined = current.join("\n").trim().to_string();
                if !joined.is_empty() {
                    parts.push(joined);
                }
                current.clear();
            }
        } else {
            current.push(line);
        }
    }
    if !current.is_empty() {
        let joined = current.join("\n").trim().to_string();
        if !joined.is_empty() {
            parts.push(joined);
        }
    }
    if parts.len() <= 1 {
        parts = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect();
    }
    parts
}

/// Edit distance, saturating at two.
///
/// Two is as far as the caller ever looks, so counting past it would be work
/// nobody reads.
#[must_use]
pub fn edits(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len().abs_diff(b.len()) > 1 {
        return 2;
    }
    let (short, long) = if a.len() > b.len() { (b, a) } else { (a, b) };
    if short.len() == long.len() {
        return short.iter().zip(long).filter(|(x, y)| x != y).count();
    }
    let (mut i, mut j, mut diffs) = (0usize, 0usize, 0usize);
    while i < short.len() && j < long.len() {
        if short[i] == long[j] {
            i += 1;
            j += 1;
            continue;
        }
        diffs += 1;
        j += 1;
        if diffs > 1 {
            return diffs;
        }
    }
    diffs + (long.len() - j)
}

/// How well one query token answers to one text token.
#[must_use]
pub fn token_score(query: &str, hay: &str) -> f64 {
    if query.is_empty() || hay.is_empty() {
        return 0.0;
    }
    if hay == query {
        return 4.0;
    }
    // "pr" is not a prefix of "prefers": a short query would match half the
    // pack on prefix alone, so it has to be exact.
    if query.len() < SHORT_EXACT {
        return 0.0;
    }
    if hay.starts_with(query) {
        return 3.0;
    }
    if hay.contains(query) {
        return 2.0;
    }
    if edits(query, hay) <= 1 {
        return 1.5;
    }
    0.0
}

/// Half-life decay on a timestamp. A missing one counts as current.
#[must_use]
pub fn recency(ts: Option<&str>, now: &str) -> f64 {
    let Some(ts) = ts.filter(|t| !t.is_empty()) else {
        return 1.0;
    };
    if clock::parse_millis(ts).is_none() {
        return 1.0;
    }
    let days = clock::elapsed_days(ts, now);
    0.5f64.powf(days / RECENCY_HALF_LIFE_DAYS)
}

/// The best each query token can do against the text, summed.
#[must_use]
pub fn text_score(query_tokens: &[String], text: &str) -> f64 {
    let hay = tokens(text);
    if hay.is_empty() {
        return 0.0;
    }
    query_tokens
        .iter()
        .map(|q| hay.iter().map(|h| token_score(q, h)).fold(0.0f64, f64::max))
        .sum()
}

fn trust_of(atom: &Record) -> f64 {
    match atom.get("trust") {
        None | Some(Value::Null) => 1.0,
        Some(other) => other.as_f64().unwrap_or(1.0),
    }
}

fn atom_in_set(atom: &Record, set: Option<&str>) -> bool {
    match set {
        None => true,
        Some(name) => atom.get("set").and_then(Value::as_str) == Some(name),
    }
}

fn file_hits(field: &str, text: &str, qtoks: &[String], bias: f64) -> Vec<Value> {
    paragraphs(text)
        .into_iter()
        .filter_map(|para| {
            let score = text_score(qtoks, &para);
            (score != 0.0).then(|| {
                json!({
                    "field": field,
                    "id": Value::Null,
                    "kind": field,
                    "text": para,
                    "score": score + bias,
                })
            })
        })
        .collect()
}

/// Sort by score descending, then field, then id, so the order is total.
fn sort_hits(hits: &mut [Value]) {
    hits.sort_by(|a, b| {
        let sa = a["score"].as_f64().unwrap_or(0.0);
        let sb = b["score"].as_f64().unwrap_or(0.0);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a["field"]
                    .as_str()
                    .unwrap_or("")
                    .cmp(b["field"].as_str().unwrap_or(""))
            })
            .then_with(|| {
                a["id"]
                    .as_str()
                    .unwrap_or("")
                    .cmp(b["id"].as_str().unwrap_or(""))
            })
    });
}

/// The prefix-and-one-edit scan over a whole pack.
#[must_use]
pub fn search_linear(
    user: &str,
    memory: &str,
    atoms: &[Record],
    query: &str,
    limit: usize,
    set: Option<&str>,
    now: &str,
) -> Vec<Value> {
    let qtoks = tokens(query);
    if qtoks.is_empty() {
        return Vec::new();
    }
    // The seat card outranks the workspace card at equal relevance, because a
    // standing preference applies wherever the question came from.
    let mut hits = file_hits("user", user, &qtoks, 0.5);
    hits.extend(file_hits("memory", memory, &qtoks, 0.25));

    for atom in atoms {
        if !record::is_live(atom, now) || !atom_in_set(atom, set) {
            continue;
        }
        let entities: Vec<String> = atom
            .get("entities")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(|i| i.as_str().map_or_else(|| i.to_string(), str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let text = atom.get("text").and_then(Value::as_str).unwrap_or("");
        let blob = format!("{text} {}", entities.join(" "));
        let relevance = text_score(&qtoks, &blob);
        let due = record::is_due(atom, now);
        if relevance == 0.0 && !due {
            continue;
        }
        let ts = atom.get("ts").and_then(Value::as_str);
        hits.push(json!({
            "field": "atom",
            "id": atom.get("id").cloned().unwrap_or(Value::Null),
            "kind": atom.get("kind").cloned().unwrap_or(Value::Null),
            "text": text,
            "due_at": atom.get("due_at").cloned().unwrap_or(Value::Null),
            "score": relevance + 0.1 * trust_of(atom) + recency(ts, now)
                + if due { 2.0 } else { 0.0 },
        }));
    }
    sort_hits(&mut hits);
    hits.truncate(limit);
    hits
}

/// Live atoms whose review is due, whatever the query says.
///
/// A review that is late is the one thing in the pack with a deadline, so it
/// leads the answer rather than competing for a place in it.
#[must_use]
pub fn due_hits(atoms: &[Record], set: Option<&str>, now: &str) -> Vec<Value> {
    let mut hits: Vec<Value> = atoms
        .iter()
        .filter(|atom| {
            record::is_live(atom, now) && atom_in_set(atom, set) && record::is_due(atom, now)
        })
        .map(|atom| {
            let ts = atom.get("ts").and_then(Value::as_str);
            json!({
                "field": "atom",
                "id": atom.get("id").cloned().unwrap_or(Value::Null),
                "kind": atom.get("kind").cloned().unwrap_or(Value::Null),
                "text": atom.get("text").and_then(Value::as_str).unwrap_or(""),
                "due_at": atom.get("due_at").cloned().unwrap_or(Value::Null),
                "score": 2.0 + 0.1 * trust_of(atom) + recency(ts, now),
            })
        })
        .collect();
    hits.sort_by(|a, b| {
        let sa = a["score"].as_f64().unwrap_or(0.0);
        let sb = b["score"].as_f64().unwrap_or(0.0);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a["id"]
                    .as_str()
                    .unwrap_or("")
                    .cmp(b["id"].as_str().unwrap_or(""))
            })
    });
    hits
}

fn hit_key(hit: &Value) -> (String, String) {
    (
        hit["field"].as_str().unwrap_or("").to_string(),
        hit["id"].as_str().unwrap_or("").to_string(),
    )
}

/// Put the due hits first, then the ranked ones, dropping repeats.
#[must_use]
pub fn front_due(due: Vec<Value>, ranked: Vec<Value>, limit: usize) -> Vec<Value> {
    if limit == 0 {
        return Vec::new();
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for hit in due.into_iter().chain(ranked) {
        if !seen.insert(hit_key(&hit)) {
            continue;
        }
        out.push(hit);
        if out.len() >= limit {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: &str = "2026-01-01T00:00:00.000Z";

    fn atom(value: Value) -> Record {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn stopwords_leave_the_query() {
        assert_eq!(tokens("what is the parser"), vec!["parser".to_string()]);
        assert!(tokens("the and of").is_empty());
    }

    #[test]
    fn a_one_block_card_splits_on_lines() {
        // Otherwise a card of one-line claims scores as one paragraph and the
        // whole file comes back as a single hit.
        let lines = paragraphs("first claim\nsecond claim\nthird claim\n");
        assert_eq!(lines.len(), 3, "{lines:?}");
        let paras = paragraphs("first block\nstill first\n\nsecond block\n");
        assert_eq!(paras.len(), 2, "{paras:?}");
    }

    #[test]
    fn edit_distance_stops_counting_at_two() {
        assert_eq!(edits("abc", "abc"), 0);
        assert_eq!(edits("abc", "abd"), 1);
        assert_eq!(edits("abc", "abcd"), 1);
        assert_eq!(edits("abc", "xyz"), 3);
        assert_eq!(
            edits("a", "abcdef"),
            2,
            "a far pair is not measured exactly"
        );
    }

    #[test]
    fn a_short_query_is_exact_only() {
        // "pr" is not a prefix of "prefers", or every query matches everything.
        assert_eq!(token_score("pr", "prefers"), 0.0);
        assert_eq!(token_score("pr", "pr"), 4.0);
        assert_eq!(token_score("pref", "prefers"), 3.0);
        assert_eq!(token_score("efer", "prefers"), 2.0);
        assert_eq!(token_score("prefer", "prefers"), 3.0);
        assert_eq!(token_score("prefrs", "prefers"), 1.5, "one edit");
    }

    #[test]
    fn recency_halves_over_the_half_life() {
        let two_weeks_ago = "2025-12-18T00:00:00.000Z";
        let weight = recency(Some(two_weeks_ago), NOW);
        assert!((weight - 0.5).abs() < 1e-9, "{weight}");
        assert_eq!(recency(None, NOW), 1.0);
        assert_eq!(recency(Some(""), NOW), 1.0);
        assert_eq!(
            recency(Some("not a stamp"), NOW),
            1.0,
            "unparsed is current"
        );
    }

    #[test]
    fn the_seat_card_outranks_the_workspace_card_at_equal_relevance() {
        let hits = search_linear(
            "prefer ripgrep",
            "prefer ripgrep",
            &[],
            "ripgrep",
            10,
            None,
            NOW,
        );
        assert_eq!(hits[0]["field"], json!("user"), "{hits:?}");
        assert!(
            hits[0]["score"].as_f64().unwrap() > hits[1]["score"].as_f64().unwrap(),
            "{hits:?}"
        );
    }

    #[test]
    fn an_atom_matches_on_its_entities_as_well_as_its_text() {
        let atoms = vec![atom(json!({
            "id": "a", "text": "nothing in the prose", "kind": "voice",
            "entities": ["ripgrep"]
        }))];
        let hits = search_linear("", "", &atoms, "ripgrep", 10, None, NOW);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0]["id"], json!("a"));
    }

    #[test]
    fn a_due_atom_is_returned_even_with_no_overlap() {
        let atoms = vec![atom(json!({
            "id": "a", "text": "utterly unrelated", "kind": "voice",
            "due_at": "2020-01-01T00:00:00.000Z"
        }))];
        let hits = search_linear("", "", &atoms, "ripgrep", 10, None, NOW);
        assert_eq!(hits.len(), 1, "a deadline outranks relevance: {hits:?}");
    }

    #[test]
    fn an_expired_atom_is_not_searched() {
        let atoms = vec![atom(json!({
            "id": "a", "text": "ripgrep here", "kind": "voice",
            "valid_to": "2020-01-01T00:00:00.000Z"
        }))];
        assert!(search_linear("", "", &atoms, "ripgrep", 10, None, NOW).is_empty());
    }

    #[test]
    fn a_set_scope_filters_the_atoms_and_not_the_cards() {
        let atoms = vec![
            atom(json!({"id": "in", "text": "ripgrep", "kind": "voice", "set": "review"})),
            atom(json!({"id": "out", "text": "ripgrep", "kind": "voice"})),
        ];
        let hits = search_linear("", "", &atoms, "ripgrep", 10, Some("review"), NOW);
        let ids: Vec<&str> = hits.iter().filter_map(|h| h["id"].as_str()).collect();
        assert_eq!(ids, vec!["in"], "{hits:?}");
    }

    #[test]
    fn an_empty_query_finds_nothing_rather_than_everything() {
        let atoms = vec![atom(json!({"id": "a", "text": "x", "kind": "voice"}))];
        assert!(search_linear("u", "m", &atoms, "", 10, None, NOW).is_empty());
        assert!(search_linear("u", "m", &atoms, "the and of", 10, None, NOW).is_empty());
    }

    #[test]
    fn due_hits_lead_and_are_not_repeated_behind_themselves() {
        let atoms = vec![atom(json!({
            "id": "due", "text": "ripgrep", "kind": "voice",
            "due_at": "2020-01-01T00:00:00.000Z"
        }))];
        let due = due_hits(&atoms, None, NOW);
        let ranked = search_linear("", "", &atoms, "ripgrep", 10, None, NOW);
        assert_eq!(due.len(), 1);
        assert_eq!(ranked.len(), 1);
        let merged = front_due(due, ranked, 10);
        assert_eq!(merged.len(), 1, "one atom, one hit: {merged:?}");
    }

    #[test]
    fn a_zero_limit_answers_nothing() {
        let atoms = vec![atom(json!({"id": "a", "text": "ripgrep", "kind": "voice"}))];
        assert!(search_linear("", "", &atoms, "ripgrep", 0, None, NOW).is_empty());
        assert!(front_due(vec![json!({})], vec![], 0).is_empty());
    }
}
