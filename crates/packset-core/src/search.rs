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
/// Whether the lexical path folds a word to its stem.
///
/// On by default, because a lexical retriever that does not stem is one that
/// misses a question asking about a wedding on a text that says weddings, and
/// every serious implementation of this scorer stems. `PACKSET_STEM=off` turns
/// it back off for a corpus where the trade goes the other way.
///
/// It is a trade. Stemming buys recall by conflating forms, and a pack holds
/// short written claims where two atoms may differ deliberately in a way a
/// stemmer erases. The default is measured rather than assumed; see the
/// retrieval section of the README for which way it went and on what.
fn stemming() -> bool {
    !matches!(
        std::env::var("PACKSET_STEM")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "off" | "0" | "no" | "false"
    )
}

/// The stemmer, built once. Snowball's English, which is Porter's suffix
/// stripping as its author later revised it (doi:10.1108/eb046814).
fn stemmer() -> &'static rust_stemmers::Stemmer {
    static ENGLISH: std::sync::OnceLock<rust_stemmers::Stemmer> = std::sync::OnceLock::new();
    ENGLISH.get_or_init(|| rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English))
}

/// Fold one token to the form the index and the query agree on.
///
/// Applied to both sides or neither: a stemmed index searched with unstemmed
/// terms matches less than no stemming at all, which is the way this is
/// usually got wrong.
#[must_use]
pub fn fold(token: &str) -> String {
    if !stemming() {
        return token.to_string();
    }
    // A token carrying a digit or a hyphen is an identifier, a version or an
    // accession rather than a word, and suffix stripping on one of those turns
    // two distinct names into one.
    if token
        .bytes()
        .any(|b| b.is_ascii_digit() || b == b'-' || b == b'_')
    {
        return token.to_string();
    }
    stemmer().stem(token).into_owned()
}

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
            // Stopwords are dropped before folding, because the list is of
            // words as written and a stemmer would not leave them matching it.
            if !STOP.contains(&token) {
                out.push(fold(token));
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

/// What a search is asked, apart from how it is scored.
///
/// Two scorers take the same question, so it is one value rather than seven
/// arguments repeated at every call: two strings of the same type next to each
/// other, then two more, is a swap nobody reads at the call site.
#[derive(Debug, Clone, Copy)]
pub struct Ask<'a> {
    /// The seat card.
    pub user: &'a str,
    /// The workspace card.
    pub memory: &'a str,
    /// The live atoms to score.
    pub atoms: &'a [Record],
    /// What was asked.
    pub query: &'a str,
    /// How many hits to return.
    pub limit: usize,
    /// The named set to stay inside, if any.
    pub set: Option<&'a str>,
    /// The instant liveness and recency are measured against.
    pub now: &'a str,
}

/// The prefix-and-one-edit scan over a whole pack.
#[must_use]
pub fn search_linear(ask: &Ask<'_>) -> Vec<Value> {
    let Ask {
        user,
        memory,
        atoms,
        query,
        limit,
        set,
        now,
    } = *ask;
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

/// The tokens one atom is searchable by: its text and its entities.
///
/// One definition, because the corpus that counts terms and the document that
/// is scored against those counts have to agree about what a document is.
#[must_use]
pub fn atom_tokens(atom: &Record) -> Vec<String> {
    let text = atom.get("text").and_then(Value::as_str).unwrap_or("");
    let mut out = tokens(text);
    if let Some(Value::Array(items)) = atom.get("entities") {
        for item in items {
            let name = item
                .as_str()
                .map_or_else(|| item.to_string(), str::to_string);
            out.extend(tokens(&name));
        }
    }
    out
}

/// The same pack, scored by BM25 over an index rather than by a scan.
///
/// A second ballot, not a replacement: the two scorers are strong at different
/// queries. The pack's own scorer finds an atom through a typo or a prefix and
/// weighs every word alike; BM25 weighs a word by how much it narrows the pack
/// down and normalises for length, and finds nothing a typo hides. The panel is
/// what turns two rankings into one.
///
/// `index` must have been built over `ask.atoms` in that order, which is what
/// lets a score name what was scored without a second lookup.
///
/// Cards are scored against the same corpus, because a standing preference
/// competes with a conclusion for the same place in the answer.
#[must_use]
pub fn search_bm25(ask: &Ask<'_>, index: &crate::bm25::Index) -> Vec<Value> {
    search_lexical(ask, index, crate::bm25::Scorer::default())
}

/// The same, in the scoring family the caller names.
///
/// One lexical ballot is a formula, not an opinion. BM25, BM25+ and query
/// likelihood disagree about different questions, which is the reason the
/// panel exists, and the panel had never been given two lexical ballots to
/// fuse because there had only ever been one lexical scorer.
#[must_use]
pub fn search_lexical(
    ask: &Ask<'_>,
    index: &crate::bm25::Index,
    scorer: crate::bm25::Scorer,
) -> Vec<Value> {
    let qtoks = tokens(ask.query);
    // Each word asked for once, which is the unweighted query written as a
    // weighted one so both paths score through the same code.
    let weighted: Vec<(String, f64)> = qtoks.into_iter().map(|term| (term, 1.0)).collect();
    bm25_hits(ask, index, &weighted, scorer)
}

/// How many of the first pass's hits the relevance model is estimated from.
const RM3_DOCS: usize = 10;

/// How many terms the model contributes.
const RM3_TERMS: usize = 10;

/// How much of the expanded query stays the words asked for.
const RM3_ALPHA: f64 = 0.5;

/// BM25 with the query expanded from its own first pass.
///
/// `documents` must be the tokenised corpus the index was built over, in that
/// order, because the relevance model is estimated from the text of the
/// documents the first pass returned rather than from the postings.
///
/// Two scoring passes instead of one, no model and nothing stored. See
/// [`crate::bm25::Index::expand`] for what is being estimated and why the
/// original query keeps a share of the weight.
#[must_use]
pub fn search_bm25_expanded(
    ask: &Ask<'_>,
    index: &crate::bm25::Index,
    documents: &[Vec<String>],
) -> Vec<Value> {
    let qtoks = tokens(ask.query);
    if qtoks.is_empty() || index.is_empty() {
        return Vec::new();
    }
    let mut first = index.score(&qtoks);
    first.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    first.truncate(RM3_DOCS);
    let feedback: Vec<(&[String], f64)> = first
        .iter()
        .filter_map(|(ordinal, score)| {
            documents
                .get(*ordinal)
                .map(|tokens| (tokens.as_slice(), *score))
        })
        .collect();
    let expanded = index.expand(&qtoks, &feedback, RM3_TERMS, RM3_ALPHA);
    bm25_hits(ask, index, &expanded, crate::bm25::Scorer::default())
}

/// The hits a weighted query scores, cards and atoms together.
fn bm25_hits(
    ask: &Ask<'_>,
    index: &crate::bm25::Index,
    query: &[(String, f64)],
    scorer: crate::bm25::Scorer,
) -> Vec<Value> {
    let Ask {
        user,
        memory,
        atoms,
        query: _,
        limit,
        set,
        now,
    } = *ask;
    if query.is_empty() || index.is_empty() {
        return Vec::new();
    }

    let mut hits: Vec<Value> = Vec::new();
    for (field, text, bias) in [("user", user, 0.5), ("memory", memory, 0.25)] {
        for para in paragraphs(text) {
            let relevance = index.score_foreign_weighted_by(scorer, query, &tokens(&para));
            if relevance == 0.0 {
                continue;
            }
            hits.push(json!({
                "field": field,
                "id": Value::Null,
                "kind": field,
                "text": para,
                "score": relevance + bias,
            }));
        }
    }

    // Only the atoms carrying a query term, straight from the postings.
    for (ordinal, relevance) in index.score_weighted_by(scorer, query) {
        let Some(atom) = atoms.get(ordinal) else {
            continue;
        };
        if !record::is_live(atom, now) || !atom_in_set(atom, set) {
            continue;
        }
        let ts = atom.get("ts").and_then(Value::as_str);
        hits.push(json!({
            "field": "atom",
            "id": atom.get("id").cloned().unwrap_or(Value::Null),
            "kind": atom.get("kind").cloned().unwrap_or(Value::Null),
            "text": atom.get("text").cloned().unwrap_or(Value::Null),
            "due_at": atom.get("due_at").cloned().unwrap_or(Value::Null),
            "score": relevance + 0.1 * trust_of(atom) + recency(ts, now),
        }));
    }

    sort_hits(&mut hits);
    hits.truncate(limit);
    hits
}

/// The weight two texts share, over the terms a model says they are about.
///
/// Both sides ascend by index, so this is one pass rather than a lookup per
/// term. A learned sparse representation is BM25's shape with the weights
/// learned instead of counted: a term the model thinks the text is about
/// carries weight even where the text says it once, and a term it thinks is
/// filler carries little where the text repeats it.
#[must_use]
pub fn sparse_dot(left: &[(u32, f32)], right: &[(u32, f32)]) -> f64 {
    let (mut here, mut there) = (0usize, 0usize);
    let mut total = 0.0f64;
    while here < left.len() && there < right.len() {
        match left[here].0.cmp(&right[there].0) {
            std::cmp::Ordering::Less => here += 1,
            std::cmp::Ordering::Greater => there += 1,
            std::cmp::Ordering::Equal => {
                total += f64::from(left[here].1) * f64::from(right[there].1);
                here += 1;
                there += 1;
            }
        }
    }
    total
}

/// Cosine between two vectors, zero when either says nothing.
///
/// Normalised here rather than assumed: the encoder normalises its output and
/// a stored vector may predate that, so dividing by the norms costs two passes
/// and removes a silent way for one atom to outrank another by magnitude.
#[must_use]
pub fn cosine(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (a, b) in left.iter().zip(right) {
        dot += f64::from(*a) * f64::from(*b);
        left_norm += f64::from(*a) * f64::from(*a);
        right_norm += f64::from(*b) * f64::from(*b);
    }
    if left_norm <= 0.0 || right_norm <= 0.0 {
        return 0.0;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

/// The vector an atom carries, if it carries one.
#[must_use]
pub fn embedding_of(atom: &Record) -> Option<Vec<f32>> {
    let items = atom.get("embedding")?.as_array()?;
    let vector: Vec<f32> = items
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();
    (vector.len() == items.len() && !vector.is_empty()).then_some(vector)
}

/// Late interaction: every query token against every document token, best wins.
///
/// A pooled vector asks whether two texts are about the same thing overall. This
/// asks whether each thing the question names is answered somewhere in the
/// document, and sums those answers, which is why it finds a short passage
/// inside a long one that pooling averages away.
///
/// Zero when either side has no tokens, so a caller can drop a document without
/// a second pass.
#[must_use]
pub fn max_sim(query: &[Vec<f32>], document: &[Vec<f32>]) -> f64 {
    if query.is_empty() || document.is_empty() {
        return 0.0;
    }
    query
        .iter()
        .map(|term| {
            document
                .iter()
                .map(|token| cosine(term, token))
                .fold(f64::MIN, f64::max)
        })
        .filter(|best| *best > f64::MIN)
        .sum()
}

/// The pack ranked by what an atom means rather than which words it used.
///
/// A third ballot. The two lexical scorers both need the question and the atom
/// to share words; this one does not, which is the whole point and also its
/// cost, since it will happily rank something adjacent above something exact.
///
/// Atoms without a stored vector are skipped rather than scored as zero: an
/// unencoded atom has not been judged irrelevant, and putting it at the bottom
/// of this ballot would let the fuse read a missing encoder as a vote.
#[must_use]
pub fn search_dense(ask: &Ask<'_>, query: &[f32]) -> Vec<Value> {
    let Ask {
        atoms,
        limit,
        set,
        now,
        ..
    } = *ask;
    if query.is_empty() {
        return Vec::new();
    }
    let mut hits: Vec<Value> = Vec::new();
    for atom in atoms {
        if !record::is_live(atom, now) || !atom_in_set(atom, set) {
            continue;
        }
        let Some(vector) = embedding_of(atom) else {
            continue;
        };
        let relevance = cosine(query, &vector);
        if relevance <= 0.0 {
            continue;
        }
        let ts = atom.get("ts").and_then(Value::as_str);
        hits.push(json!({
            "field": "atom",
            "id": atom.get("id").cloned().unwrap_or(Value::Null),
            "kind": atom.get("kind").cloned().unwrap_or(Value::Null),
            "text": atom.get("text").cloned().unwrap_or(Value::Null),
            "due_at": atom.get("due_at").cloned().unwrap_or(Value::Null),
            "score": relevance + 0.1 * trust_of(atom) + recency(ts, now),
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

    /// The point of late interaction: a question whose terms are answered in
    /// different parts of one document, which pooling averages away.
    #[test]
    fn late_interaction_sums_the_best_match_for_each_query_term() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let c = vec![0.0f32, 0.0, 1.0];
        // Both query terms are matched exactly, in different tokens.
        let scored = max_sim(&[a.clone(), b.clone()], &[c.clone(), a.clone(), b.clone()]);
        assert!((scored - 2.0).abs() < 1e-9, "{scored}");
        // One matched, one absent.
        let half = max_sim(&[a.clone(), b.clone()], &[a.clone(), c.clone()]);
        assert!(half < scored, "{half} vs {scored}");
        assert_eq!(max_sim(&[], std::slice::from_ref(&a)), 0.0);
        assert_eq!(max_sim(std::slice::from_ref(&a), &[]), 0.0);
    }

    /// A hit with a key and a score, for the fusion tests.
    fn scored(id: &str, score: f64) -> Value {
        json!({ "field": "atom", "id": id, "text": id, "score": score })
    }

    /// A question about a wedding finds a text that says weddings, which is
    /// the whole reason a lexical retriever stems.
    #[test]
    fn a_question_finds_the_other_form_of_the_word() {
        let asked = tokens("what did she say about the wedding");
        let said = tokens("we talked about weddings and rings");
        assert!(
            asked.iter().any(|t| said.contains(t)),
            "{asked:?} shares nothing with {said:?}"
        );
    }

    /// Both sides fold or neither. A stemmed index searched with unstemmed
    /// terms matches less than no stemming at all, and that is the usual way
    /// this is got wrong.
    #[test]
    fn the_index_and_the_query_fold_the_same_way() {
        let index = atom_tokens(
            json!({ "text": "the parser reads manifests", "kind": "conclusion" })
                .as_object()
                .expect("object"),
        );
        for term in tokens("which manifest does the parser read") {
            if term == "manifest" || term == "parser" || term == "read" {
                assert!(index.contains(&term), "{term} is not in {index:?}");
            }
        }
    }

    /// A stemmer is for words. An accession, a version or an identifier is a
    /// name, and stripping a suffix off one merges two distinct things.
    #[test]
    fn a_name_is_not_stemmed() {
        for name in ["deed-patch-notes", "sha256:abc", "v0_9_3", "utf8"] {
            for token in tokens(name) {
                assert_eq!(fold(&token), token, "{name} was folded through {token}");
            }
        }
    }

    /// Only the terms both sides carry count, and the pass depends on both
    /// sides ascending.
    #[test]
    fn shared_terms_are_the_only_ones_that_count() {
        let left = [(1u32, 0.5f32), (4, 2.0), (9, 1.0)];
        let right = [(2u32, 3.0f32), (4, 0.5), (9, 0.25)];
        // 4 and 9 are shared: 2.0*0.5 + 1.0*0.25.
        assert!((sparse_dot(&left, &right) - 1.25).abs() < 1e-9);
        // Nothing shared is nothing, not an error and not a default score.
        assert_eq!(sparse_dot(&left, &[(2u32, 1.0f32), (3, 1.0)]), 0.0);
        assert_eq!(sparse_dot(&[], &right), 0.0);
    }

    /// Every name the panel accepts has to reach its own implementation.
    ///
    /// Five of the nine used to fall through to Borda, so asking for Schulze
    /// got Borda's answer under Schulze's name. The enum's own doc says only
    /// implemented names parse, and that has to be true of what runs rather
    /// than only of what parses.
    #[test]
    fn a_named_fuse_runs_the_voter_it_names() {
        // `c` is second on both ballots and close behind the leader on both,
        // so it carries the most score mass and the least rank credit. A voter
        // that reads scores puts it first; a voter that reads positions cannot.
        let first = vec![scored("a", 10.0), scored("c", 9.9), scored("b", 1.0)];
        let second = vec![scored("b", 10.0), scored("c", 9.9), scored("a", 1.0)];
        let now = crate::clock::utcnow();

        let order = |name: &str| -> Vec<String> {
            let panel = crate::panel::Panel::named(name, "none", "off").expect("voter");
            merge_ballots(&[first.clone(), second.clone()], 3, &panel, &now)
                .iter()
                .map(|hit| hit["id"].as_str().unwrap_or("").to_string())
                .collect()
        };

        let borda = order("borda");
        assert_eq!(borda.first().map(String::as_str), Some("a"), "{borda:?}");

        // The bug this pins: both score fusions used to return Borda's answer.
        for name in ["combsum", "combmnz"] {
            let ranked = order(name);
            assert_eq!(
                ranked.first().map(String::as_str),
                Some("c"),
                "{name}: {ranked:?}"
            );
            assert_ne!(ranked, borda, "{name} is still answering as borda");
        }

        // And every rank voter answers with what its own module answers,
        // which is the property a fall-through breaks silently: a wrong
        // dispatch still returns a plausible full ranking.
        let lists = [first.clone(), second.clone()];
        let keys: Vec<Vec<String>> = lists.iter().map(|hits| ballot_keys(hits)).collect();
        let k = 3;
        let expected: Vec<(&str, Vec<String>)> = vec![
            ("rrf", crate::rrf::rrf_merge(&keys, RRF_K0)),
            ("dowdall", crate::dowdall::dowdall_merge(&keys, k)),
            ("kemeny", crate::kemeny::kemeny_merge(&keys, k)),
            ("schulze", crate::schulze::schulze_merge(&keys, k)),
            ("copeland", crate::copeland::copeland_merge(&keys, k)),
            ("tideman", crate::tideman::ranked_pairs_merge(&keys, k)),
        ];
        for (name, want) in expected {
            let want: Vec<String> = want
                .iter()
                .map(|key| key.rsplit('\u{0}').next().unwrap_or(key).to_string())
                .collect();
            assert_eq!(order(name), want, "{name} did not answer as its own module");
        }
    }

    /// Position stands in for a weight, so the diversify slot downstream sees a
    /// relevance that decreases with rank whichever voter produced the order.
    #[test]
    fn a_fused_weight_falls_with_position() {
        let ballots = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        let scored_ballots = vec![vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.5),
            ("c".to_string(), 0.1),
        ]];
        let (ranked, weights) =
            fuse_scores(crate::panel::Fuse::Borda, &ballots, &scored_ballots, 3);
        assert_eq!(ranked, vec!["a", "b", "c"]);
        assert!(weights["a"] > weights["b"] && weights["b"] > weights["c"]);
    }

    /// One question, for the tests that only vary part of it.
    fn ask<'a>(
        user: &'a str,
        memory: &'a str,
        atoms: &'a [Record],
        query: &'a str,
        limit: usize,
        set: Option<&'a str>,
    ) -> Ask<'a> {
        Ask {
            user,
            memory,
            atoms,
            query,
            limit,
            set,
            now: NOW,
        }
    }

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
        let hits = search_linear(&ask(
            "prefer ripgrep",
            "prefer ripgrep",
            &[],
            "ripgrep",
            10,
            None,
        ));
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
        let hits = search_linear(&ask("", "", &atoms, "ripgrep", 10, None));
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0]["id"], json!("a"));
    }

    #[test]
    fn a_due_atom_is_returned_even_with_no_overlap() {
        let atoms = vec![atom(json!({
            "id": "a", "text": "utterly unrelated", "kind": "voice",
            "due_at": "2020-01-01T00:00:00.000Z"
        }))];
        let hits = search_linear(&ask("", "", &atoms, "ripgrep", 10, None));
        assert_eq!(hits.len(), 1, "a deadline outranks relevance: {hits:?}");
    }

    #[test]
    fn an_expired_atom_is_not_searched() {
        let atoms = vec![atom(json!({
            "id": "a", "text": "ripgrep here", "kind": "voice",
            "valid_to": "2020-01-01T00:00:00.000Z"
        }))];
        assert!(search_linear(&ask("", "", &atoms, "ripgrep", 10, None)).is_empty());
    }

    #[test]
    fn a_set_scope_filters_the_atoms_and_not_the_cards() {
        let atoms = vec![
            atom(json!({"id": "in", "text": "ripgrep", "kind": "voice", "set": "review"})),
            atom(json!({"id": "out", "text": "ripgrep", "kind": "voice"})),
        ];
        let hits = search_linear(&ask("", "", &atoms, "ripgrep", 10, Some("review")));
        let ids: Vec<&str> = hits.iter().filter_map(|h| h["id"].as_str()).collect();
        assert_eq!(ids, vec!["in"], "{hits:?}");
    }

    #[test]
    fn an_empty_query_finds_nothing_rather_than_everything() {
        let atoms = vec![atom(json!({"id": "a", "text": "x", "kind": "voice"}))];
        assert!(search_linear(&ask("u", "m", &atoms, "", 10, None)).is_empty());
        assert!(search_linear(&ask("u", "m", &atoms, "the and of", 10, None)).is_empty());
    }

    #[test]
    fn due_hits_lead_and_are_not_repeated_behind_themselves() {
        let atoms = vec![atom(json!({
            "id": "due", "text": "ripgrep", "kind": "voice",
            "due_at": "2020-01-01T00:00:00.000Z"
        }))];
        let due = due_hits(&atoms, None, NOW);
        let ranked = search_linear(&ask("", "", &atoms, "ripgrep", 10, None));
        assert_eq!(due.len(), 1);
        assert_eq!(ranked.len(), 1);
        let merged = front_due(due, ranked, 10);
        assert_eq!(merged.len(), 1, "one atom, one hit: {merged:?}");
    }

    #[test]
    fn a_zero_limit_answers_nothing() {
        let atoms = vec![atom(json!({"id": "a", "text": "ripgrep", "kind": "voice"}))];
        assert!(search_linear(&ask("", "", &atoms, "ripgrep", 0, None)).is_empty());
        assert!(front_due(vec![json!({})], vec![], 0).is_empty());
    }
}

/// The identity of a hit across two ranked lists.
///
/// Field and id together, because a prose hit has no id and two of them from
/// different cards must not collapse into one.
#[must_use]
pub fn hit_key_of(hit: &Value) -> String {
    format!(
        "{}\u{0}{}",
        hit["field"].as_str().unwrap_or(""),
        hit["id"].as_str().unwrap_or("")
    )
}

/// Distinct keys of a ranked list, in order.
fn ballot_keys(hits: &[Value]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    hits.iter()
        .map(hit_key_of)
        .filter(|key| seen.insert(key.clone()))
        .collect()
}

/// Fused scores, keeping first-seen order through a tie.
///
/// A tie broken by hash order would reorder results between two runs over the
/// same data, so the order a key was first seen in decides.
/// Reciprocal rank fusion's smoothing constant, as the paper sets it.
const RRF_K0: usize = 60;

/// Run the named voter and give every key a weight from where it landed.
///
/// One dispatch, to the module that implements the name. The voters do not
/// share a score scale, so position stands in as the weight: what reads it is
/// the diversify slot, which needs a monotone relevance and not a calibrated
/// one.
fn fuse_scores(
    fuse: crate::panel::Fuse,
    ballots: &[Vec<String>],
    scored: &[crate::comb::ScoredBallot<String>],
    k: usize,
) -> (Vec<String>, std::collections::HashMap<String, f64>) {
    use crate::panel::Fuse;
    if ballots.is_empty() || k == 0 {
        return (Vec::new(), std::collections::HashMap::new());
    }
    let ranked = match fuse {
        Fuse::Borda => crate::borda::borda_merge(ballots, k),
        Fuse::Rrf => crate::rrf::rrf_merge(ballots, RRF_K0),
        // The two score fusions are the only voters that read the scores; every
        // other one reads position alone.
        Fuse::CombSum => crate::comb::combsum_merge(scored),
        Fuse::CombMnz => crate::comb::combmnz_merge(scored),
        Fuse::Dowdall => crate::dowdall::dowdall_merge(ballots, k),
        Fuse::Kemeny => crate::kemeny::kemeny_merge(ballots, k),
        Fuse::Schulze => crate::schulze::schulze_merge(ballots, k),
        Fuse::Copeland => crate::copeland::copeland_merge(ballots, k),
        Fuse::Tideman => crate::tideman::ranked_pairs_merge(ballots, k),
    };
    let n = ranked.len() as f64;
    let scores = ranked
        .iter()
        .enumerate()
        .map(|(index, key)| (key.clone(), n - index as f64))
        .collect();
    (ranked, scores)
}

/// One ballot as (key, score) pairs, which is what a score fusion needs.
fn scored_ballot(hits: &[Value]) -> crate::comb::ScoredBallot<String> {
    let mut seen = std::collections::HashSet::new();
    hits.iter()
        .filter_map(|hit| {
            let key = hit_key_of(hit);
            seen.insert(key.clone())
                .then(|| (key, hit["score"].as_f64().unwrap_or(0.0)))
        })
        .collect()
}

/// Named fuse, then diversify, then decay, over two ranked lists.
///
/// Not a de-duplicate: two lists that agree about a hit are two votes for it,
/// which is the whole reason to run a panel rather than concatenate.
#[must_use]
pub fn merge_ballots(
    ballots: &[Vec<Value>],
    limit: usize,
    panel: &crate::panel::Panel,
    now: &str,
) -> Vec<Value> {
    if limit == 0 {
        return Vec::new();
    }
    let mut by_key: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    // Reversed, so an earlier list's copy of a hit is the one kept.
    for hits in ballots.iter().rev() {
        for hit in hits {
            by_key.insert(hit_key_of(hit), hit.clone());
        }
    }
    let keys: Vec<Vec<String>> = ballots.iter().map(|hits| ballot_keys(hits)).collect();
    let scored: Vec<crate::comb::ScoredBallot<String>> =
        ballots.iter().map(|hits| scored_ballot(hits)).collect();
    let (mut ranked, scores) = fuse_scores(panel.fuse, &keys, &scored, limit);
    let mut weights: std::collections::HashMap<String, f64> = ranked
        .iter()
        .map(|key| (key.clone(), scores.get(key).copied().unwrap_or(0.0)))
        .collect();

    if panel.decay == crate::panel::Decay::On {
        let order: std::collections::HashMap<&String, usize> =
            ranked.iter().enumerate().map(|(i, k)| (k, i)).collect();
        for key in &ranked {
            let hit = &by_key[key];
            let source = hit["field"].as_str().unwrap_or("");
            let age = hit["ts"]
                .as_str()
                .map_or(0.0, |ts| clock::elapsed_days(ts, now));
            if let Some(weight) = weights.get_mut(key) {
                *weight *= panel.decay_weight(source, age);
            }
        }
        let mut sorted = ranked.clone();
        sorted.sort_by(|a, b| {
            let wa = weights[a];
            let wb = weights[b];
            wb.partial_cmp(&wa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| order[a].cmp(&order[b]))
        });
        ranked = sorted;
    }

    let items: Vec<crate::mmr::Ranked> = ranked
        .iter()
        .map(|key| crate::mmr::Ranked {
            id: key.clone(),
            rel: weights[key],
            tokens: raw_tokens(by_key[key]["text"].as_str().unwrap_or("")),
        })
        .collect();
    let order = panel.rerank(&items, 0.7);
    order
        .into_iter()
        .filter_map(|key| by_key.get(&key).cloned())
        .take(limit)
        .collect()
}

/// Tokens with the stopwords kept, which is what a similarity wants.
///
/// Dropping them here would make two hits that share nothing but "the" look
/// alike to the diversifier.
fn raw_tokens(text: &str) -> std::collections::HashSet<String> {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut out = std::collections::HashSet::new();
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
            out.insert(lower[start..i].to_string());
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod merge_tests {
    use super::*;
    use crate::panel::Panel;
    use serde_json::json;

    const NOW: &str = "2026-01-01T00:00:00.000Z";

    fn hit(field: &str, id: &str, text: &str) -> Value {
        json!({"field": field, "id": id, "text": text, "score": 1.0})
    }

    fn default_panel() -> Panel {
        Panel::named("borda", "none", "off").unwrap()
    }

    #[test]
    fn a_hit_two_lists_agree_on_outranks_one_only_a_leader_named() {
        // Two votes beat one first place, which is the reason to run a panel
        // rather than concatenate the lists.
        let a = vec![hit("atom", "solo", "alpha"), hit("atom", "both", "beta")];
        let b = vec![hit("atom", "both", "beta"), hit("atom", "other", "gamma")];
        let merged = merge_ballots(&[a, b], 10, &default_panel(), NOW);
        assert_eq!(merged[0]["id"], json!("both"), "{merged:?}");
    }

    #[test]
    fn a_prose_hit_with_no_id_does_not_collapse_into_another() {
        let a = vec![hit("user", "", "one"), hit("memory", "", "two")];
        let merged = merge_ballots(&[a], 10, &default_panel(), NOW);
        assert_eq!(merged.len(), 2, "{merged:?}");
    }

    #[test]
    fn a_tie_keeps_the_order_it_was_first_seen_in() {
        // Otherwise two runs over the same data disagree.
        let a = vec![hit("atom", "x", "one"), hit("atom", "y", "two")];
        let b = vec![hit("atom", "y", "two"), hit("atom", "x", "one")];
        let first = merge_ballots(&[a.clone(), b.clone()], 10, &default_panel(), NOW);
        let second = merge_ballots(&[a, b], 10, &default_panel(), NOW);
        assert_eq!(first, second);
        assert_eq!(first[0]["id"], json!("x"), "{first:?}");
    }

    #[test]
    fn the_limit_is_applied_after_the_merge() {
        let a = vec![hit("atom", "a", "one"), hit("atom", "b", "two")];
        let b = vec![hit("atom", "c", "three")];
        assert_eq!(
            merge_ballots(&[a.clone(), b.clone()], 1, &default_panel(), NOW).len(),
            1
        );
        assert!(merge_ballots(&[a, b], 0, &default_panel(), NOW).is_empty());
    }

    #[test]
    fn an_empty_ballot_does_not_erase_the_other() {
        let a = vec![hit("atom", "a", "one")];
        let merged = merge_ballots(&[a, Vec::new()], 10, &default_panel(), NOW);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn the_diversifier_reads_stopwords_too() {
        // "the" and "of" carry no query signal but they do say two texts look
        // alike, which is a different question.
        let with = raw_tokens("the parser of the header");
        assert!(with.contains("the"), "{with:?}");
        assert!(!tokens("the parser of the header").contains(&"the".to_string()));
    }
}
