//! One atom as it sits in the store: a JSON object.
//!
//! The record is a `serde_json` object rather than a struct because the store
//! is shared with a writer that may carry fields this one does not model, and
//! dropping a field on a round trip would lose somebody's data. Every rule
//! here reads the fields it knows and leaves the rest alone.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::atom::{check_entity, EntityRefusal};
use crate::clock;
use crate::prose;

/// Schema name for a pack atom.
pub const SCHEMA: &str = crate::atom::SCHEMA;
/// Characters allowed in `USER.md`.
pub const USER_CAP: usize = 1375;
/// Characters allowed in a workspace `MEMORY.md`.
pub const MEMORY_CAP: usize = 2200;
/// Characters allowed in one atom's text.
pub const TEXT_SOFT_CAP: usize = 500;
/// Jaccard at or above which two atoms link.
pub const LINK_THRESHOLD: f64 = 0.3;
/// Review interval when nothing has been graded yet.
pub const DEFAULT_REVIEW_INTERVAL_S: i64 = 86_400;
/// SM-2 style ease, kept for readers of the review block.
pub const REVIEW_EASE: f64 = 2.5;
/// Starting stability, in days.
pub const DEFAULT_STABILITY: f64 = 1.0;
/// Starting difficulty, on a one to ten scale.
pub const DEFAULT_DIFFICULTY: f64 = 5.0;

/// What an atom may claim to be.
pub const KINDS: &[&str] = &[
    "voice",
    "habit",
    "cache-pointer",
    "preference",
    "lesson",
    "goal",
    "conclusion",
    "card_line",
    "summary",
    "correction",
    "belief",
];

/// Whether the claim was stated or inferred.
pub const LEVELS: &[&str] = &["explicit", "derived"];

/// A string the way Python's `repr` writes it.
///
/// The error messages cross the wire to clients that already read them, so
/// quoting a set name or an entity differently would be a change in the API
/// rather than in the prose. Python prefers single quotes and switches to
/// double only when the value carries a single quote and no double.
#[must_use]
pub fn py_repr(value: &str) -> String {
    let has_single = value.contains('\'');
    let has_double = value.contains('"');
    let quote = if has_single && !has_double { '"' } else { '\'' };
    let mut out = String::with_capacity(value.len() + 2);
    out.push(quote);
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Why a record cannot be stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomError(pub String);

impl std::fmt::Display for AtomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AtomError {}

impl From<prose::ProseError> for AtomError {
    fn from(err: prose::ProseError) -> Self {
        Self(err.0)
    }
}

/// Zero-width and bidirectional controls, which hide text from a reader.
fn has_invisible(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c as u32,
            0x200b..=0x200f | 0x202a..=0x202e | 0x2060..=0x206f | 0xfeff)
    })
}

const SECRET_KEYS: &[&str] = &[
    "api_key", "api-key", "apikey", "secret", "password", "token",
];

/// Whether the text carries something credential-shaped.
///
/// A key assigned to a name, a bearer token, or the `sk-` prefix the common
/// model APIs mint.
/// The pack is pasted into a model's context, so this is the one place the
/// paste is automatic rather than a person deciding.
#[must_use]
pub fn looks_like_a_secret(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    for key in SECRET_KEYS {
        let mut from = 0usize;
        while let Some(at) = lower[from..].find(key) {
            let idx = from + at;
            let before_is_word = idx
                .checked_sub(1)
                .and_then(|i| lower.as_bytes().get(i))
                .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-');
            let after = lower[idx + key.len()..].trim_start();
            if !before_is_word && (after.starts_with('=') || after.starts_with(':')) {
                return true;
            }
            from = idx + key.len();
        }
    }
    if let Some(at) = lower.find("bearer ") {
        let rest = lower[at + 7..].trim_start();
        let run = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
            .count();
        if run >= 8 {
            return true;
        }
    }
    let mut from = 0usize;
    while let Some(at) = lower[from..].find("sk-") {
        let idx = from + at;
        let run = lower[idx + 3..]
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .count();
        if run >= 8 {
            return true;
        }
        from = idx + 3;
    }
    false
}

/// Refuse text a reader cannot see or should never have been handed.
///
/// # Errors
///
/// Returns [`AtomError`] for invisible unicode or credential-shaped text.
pub fn reject_unsafe(text: &str) -> Result<(), AtomError> {
    if has_invisible(text) {
        return Err(AtomError("invisible unicode is rejected".into()));
    }
    if looks_like_a_secret(text) {
        return Err(AtomError("credential-shaped text is rejected".into()));
    }
    Ok(())
}

fn refusal_message(value: &str, why: EntityRefusal) -> String {
    match why {
        EntityRefusal::Empty => "an entity cannot be empty".to_string(),
        EntityRefusal::BarePrefix => {
            let prefix = crate::atom::ACCESSION_PREFIXES
                .iter()
                .find(|p| value.trim().starts_with(**p))
                .copied()
                .unwrap_or("");
            format!(
                "{} is a bare {} and names no deed",
                py_repr(value),
                py_repr(prefix)
            )
        }
        EntityRefusal::Separator(bad) => {
            format!(
                "{} carries {}, which would split the entity into two",
                py_repr(value),
                py_repr(&bad.to_string())
            )
        }
    }
}

/// Check a record and normalise the fields that have one legal form.
///
/// # Errors
///
/// Returns [`AtomError`] for an unknown kind or level, missing text or
/// workspace, text past the soft cap, a bad set name, an entity that opens like
/// an accession and is not one, or prose too complex for one claim.
pub fn validate(atom: &mut Map<String, Value>) -> Result<(), AtomError> {
    let kind = atom.get("kind").and_then(Value::as_str).unwrap_or("");
    if !KINDS.contains(&kind) {
        let shown = atom.get("kind").map_or("None".into(), value_repr);
        return Err(AtomError(format!("unknown atom kind: {shown}")));
    }
    let level = atom
        .get("level")
        .and_then(Value::as_str)
        .unwrap_or("explicit");
    if !LEVELS.contains(&level) {
        let shown = atom.get("level").map_or("None".into(), value_repr);
        return Err(AtomError(format!("unknown atom level: {shown}")));
    }
    let text = match atom.get("text").and_then(Value::as_str) {
        Some(t) if !t.trim().is_empty() => t.to_string(),
        _ => return Err(AtomError("atom text is required".into())),
    };
    reject_unsafe(&text)?;
    if text.chars().count() > TEXT_SOFT_CAP {
        return Err(AtomError(format!(
            "atom text exceeds soft cap {TEXT_SOFT_CAP}"
        )));
    }
    if atom
        .get("workspace")
        .and_then(Value::as_str)
        .unwrap_or("")
        .is_empty()
    {
        return Err(AtomError("atom workspace is required".into()));
    }

    match atom.get("set").and_then(Value::as_str) {
        Some(raw) if !raw.is_empty() => {
            let named = crate::set_name::check(raw).map_err(AtomError)?;
            atom.insert("set".into(), Value::String(named));
        }
        _ => {
            atom.remove("set");
        }
    }

    if let Some(raw) = atom.get("entities").cloned() {
        if let Some(items) = raw.as_array() {
            let mut checked = Vec::with_capacity(items.len());
            for item in items {
                let text = item
                    .as_str()
                    .map_or_else(|| value_text(item), str::to_string);
                match check_entity(&text) {
                    Ok(kept) => checked.push(Value::String(kept.to_string())),
                    Err(why) => return Err(AtomError(refusal_message(&text, why))),
                }
            }
            atom.insert("entities".into(), Value::Array(checked));
        }
    }

    let report = prose::refuse(&text, prose::Role::Atom)?;
    atom.insert("prose".into(), prose_value(&report));
    Ok(())
}

fn value_repr(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn prose_value(report: &prose::Report) -> Value {
    let mut out = Map::new();
    out.insert("words".into(), report.words.into());
    out.insert("sentences".into(), report.sentences.into());
    out.insert("grade".into(), number_or_null(report.grade));
    out.insert("ease".into(), number_or_null(report.ease));
    out.insert("adverbs".into(), report.adverbs.into());
    out.insert(
        "adverb_ratio".into(),
        number_or_null(Some(report.adverb_ratio)),
    );
    out.insert("passives".into(), report.passives.into());
    out.insert("hard_sentences".into(), report.hard_sentences.into());
    out.insert(
        "very_hard_sentences".into(),
        report.very_hard_sentences.into(),
    );
    Value::Object(out)
}

fn number_or_null(v: Option<f64>) -> Value {
    v.and_then(serde_json::Number::from_f64)
        .map_or(Value::Null, Value::Number)
}

/// Live set: not tombstoned, and `valid_to` missing or still open.
#[must_use]
pub fn is_live(atom: &Map<String, Value>, now: &str) -> bool {
    if atom
        .get("tombstone")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return false;
    }
    match atom.get("valid_to") {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) if s.is_empty() => true,
        Some(other) => value_text(other).as_str() > now,
    }
}

/// Review clock. A missing `due_at` is not due, and `valid_to` is not consulted.
#[must_use]
pub fn is_due(atom: &Map<String, Value>, now: &str) -> bool {
    match atom.get("due_at") {
        None | Some(Value::Null) => false,
        Some(Value::String(s)) if s.is_empty() => false,
        Some(other) => value_text(other).as_str() <= now,
    }
}

/// The names an atom is about.
///
/// A declared `entities` list wins. Without one, capitalised runs and backtick
/// names are the fallback, which is what makes an atom nobody annotated still
/// link to its neighbours.
#[must_use]
pub fn entities_of(atom: &Map<String, Value>) -> BTreeSet<String> {
    if let Some(Value::Array(items)) = atom.get("entities") {
        return items
            .iter()
            .map(|item| value_text(item).trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    let text = atom.get("text").and_then(Value::as_str).unwrap_or("");
    let mut names: BTreeSet<String> = capitalized_runs(text);
    names.extend(backtick_names(text));
    names
}

/// `\b[A-Z][A-Za-z0-9]{1,}\b`
fn capitalized_runs(text: &str) -> BTreeSet<String> {
    let bytes = text.as_bytes();
    let mut out = BTreeSet::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let boundary = i == 0 || !is_word_byte(bytes[i - 1]);
        if boundary && bytes[i].is_ascii_uppercase() {
            let start = i;
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
                i += 1;
            }
            // {1,} after the first character means at least two in total, and
            // the match must end on a word boundary.
            if i - start >= 2 && (i == bytes.len() || !is_word_byte(bytes[i])) {
                out.insert(text[start..i].to_string());
            }
        } else {
            i += 1;
        }
    }
    out
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Text between backticks, trimmed, empties dropped.
fn backtick_names(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut rest = text;
    while let Some(open) = rest.find('`') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('`') else { break };
        let inner = after[..close].trim();
        if !inner.is_empty() {
            out.insert(inner.to_string());
        }
        rest = &after[close + 1..];
    }
    out
}

/// Ids of live peers whose entity sets meet the threshold.
#[must_use]
pub fn link_targets(
    atom: &Map<String, Value>,
    peers: &[Map<String, Value>],
    threshold: f64,
    now: &str,
) -> Vec<String> {
    let mine = entities_of(atom);
    let mine_refs: Vec<&str> = mine.iter().map(String::as_str).collect();
    let atom_id = atom.get("id").and_then(Value::as_str);
    let mut linked = Vec::new();
    for other in peers {
        let other_id = other.get("id").and_then(Value::as_str);
        if other_id.is_none() || other_id == atom_id {
            continue;
        }
        if !is_live(other, now) {
            continue;
        }
        let theirs = entities_of(other);
        let theirs_refs: Vec<&str> = theirs.iter().map(String::as_str).collect();
        if crate::atom::entity_jaccard(mine_refs.iter().copied(), theirs_refs.iter().copied())
            >= threshold
        {
            linked.push(other_id.unwrap_or_default().to_string());
        }
    }
    linked
}

/// Set overlap links on `atom` and rewrite the peers that changed.
///
/// A link is symmetric, so adding one to an atom means adding it to the peer,
/// and dropping one means dropping it on both sides. The returned peers are the
/// ones the caller has to write back.
pub fn apply_links(
    atom: &mut Map<String, Value>,
    live: &[Map<String, Value>],
    threshold: f64,
    now: &str,
) -> Vec<Map<String, Value>> {
    let atom_id = atom
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let peers: Vec<Map<String, Value>> = live
        .iter()
        .filter(|other| {
            other.get("id").and_then(Value::as_str) != Some(atom_id.as_str()) && is_live(other, now)
        })
        .cloned()
        .collect();
    let targets: BTreeSet<String> = link_targets(atom, &peers, threshold, now)
        .into_iter()
        .collect();
    atom.insert(
        "links".into(),
        Value::Array(
            targets
                .iter()
                .map(|id| Value::String(id.clone()))
                .collect::<Vec<_>>(),
        ),
    );

    let mut rewritten = Vec::new();
    for mut other in peers {
        let other_id = other
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let mut links: BTreeSet<String> = other
            .get("links")
            .and_then(Value::as_array)
            .map(|items| items.iter().map(value_text).collect())
            .unwrap_or_default();
        let should_link = targets.contains(&other_id);
        let has_link = links.contains(&atom_id);
        if should_link == has_link {
            continue;
        }
        if should_link {
            links.insert(atom_id.clone());
        } else {
            links.remove(&atom_id);
        }
        other.insert(
            "links".into(),
            Value::Array(links.into_iter().map(Value::String).collect()),
        );
        rewritten.push(other);
    }
    rewritten
}

/// Drop links pointing outside the supplied live set.
pub fn filter_live_links(atoms: &mut [Map<String, Value>]) {
    let live: BTreeSet<String> = atoms
        .iter()
        .filter_map(|a| a.get("id").and_then(Value::as_str).map(str::to_string))
        .collect();
    for atom in atoms.iter_mut() {
        let kept: Vec<Value> = atom
            .get("links")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter(|item| live.contains(&value_text(item)))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        atom.insert("links".into(), Value::Array(kept));
    }
}

/// How a review turned out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grade {
    /// First scheduling, or a re-schedule that is not a review.
    Initial,
    /// The atom came back.
    Recalled,
    /// It did not.
    Lapsed,
}

/// Set `due_at` from stability and difficulty. `valid_to` is left alone.
///
/// A lapse halves stability and nudges difficulty up; a recall grows stability
/// by how overdue the atom was, so an atom that survived a long gap earns a
/// longer one. The live set and the review clock are separate questions, which
/// is why this never touches `valid_to`.
pub fn schedule_review(
    atom: &mut Map<String, Value>,
    now: &str,
    grade: Grade,
    interval_s: Option<i64>,
) {
    let previous = atom
        .get("review")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let read = |key: &str, fallback: f64| -> f64 {
        previous
            .get(key)
            .and_then(Value::as_f64)
            .filter(|v| *v != 0.0)
            .unwrap_or(fallback)
    };
    let mut stability = read("stability", DEFAULT_STABILITY);
    let mut difficulty = read("difficulty", DEFAULT_DIFFICULTY);

    let mut review = Map::new();
    let span;
    match grade {
        Grade::Lapsed => {
            difficulty = (difficulty + 0.2).clamp(1.0, 10.0);
            stability = (stability * 0.5).max(0.1);
            span = (stability.max(1.0) * 86_400.0) as i64;
            review.insert("reps".into(), 0.into());
            review.insert("interval_s".into(), span.into());
            review.insert("ease".into(), number_or_null(Some(REVIEW_EASE)));
            review.insert("stability".into(), number_or_null(Some(stability)));
            review.insert("difficulty".into(), number_or_null(Some(difficulty)));
            review.insert("last".into(), Value::String(now.to_string()));
        }
        Grade::Recalled => {
            let reps = previous
                .get("reps")
                .and_then(Value::as_i64)
                .unwrap_or(0)
                .saturating_add(1);
            let last = previous.get("last").and_then(Value::as_str).unwrap_or("");
            let elapsed = if last.is_empty() {
                0.0
            } else {
                clock::elapsed_days(last, now)
            };
            let retr = if stability > 0.0 {
                0.9f64.powf(elapsed / stability)
            } else {
                0.0
            }
            .clamp(0.01, 0.99);
            difficulty = (difficulty - 0.15).clamp(1.0, 10.0);
            stability *= 1.0 + (1.0 - difficulty / 10.0).exp() * (1.0 - retr);
            span = (stability.max(1.0) * 86_400.0) as i64;
            review.insert("reps".into(), reps.into());
            review.insert("interval_s".into(), span.into());
            review.insert("ease".into(), number_or_null(Some(REVIEW_EASE)));
            review.insert("stability".into(), number_or_null(Some(stability)));
            review.insert("difficulty".into(), number_or_null(Some(difficulty)));
            review.insert("last".into(), Value::String(now.to_string()));
        }
        Grade::Initial => {
            span = interval_s.unwrap_or(DEFAULT_REVIEW_INTERVAL_S);
            review = previous;
            review.entry("reps").or_insert_with(|| 0.into());
            review.entry("interval_s").or_insert_with(|| span.into());
            review
                .entry("ease")
                .or_insert_with(|| number_or_null(Some(REVIEW_EASE)));
            review
                .entry("stability")
                .or_insert_with(|| number_or_null(Some(DEFAULT_STABILITY)));
            review
                .entry("difficulty")
                .or_insert_with(|| number_or_null(Some(DEFAULT_DIFFICULTY)));
            review
                .entry("last")
                .or_insert_with(|| Value::String(now.to_string()));
        }
    }
    if let Some(due) = clock::shift(now, span) {
        atom.insert("due_at".into(), Value::String(due));
    }
    atom.insert("review".into(), Value::Object(review));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn atom(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn repr_prefers_single_quotes_the_way_python_does() {
        assert_eq!(py_repr("plain"), "'plain'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("say \"hi\""), "'say \"hi\"'");
        assert_eq!(py_repr("both ' and \""), "'both \\' and \"'");
        assert_eq!(py_repr("a\nb"), "'a\\nb'");
    }

    #[test]
    fn a_credential_shape_is_caught_and_a_word_containing_one_is_not() {
        assert!(looks_like_a_secret("api_key=abcd1234"));
        assert!(looks_like_a_secret("Secret: hunter2"));
        assert!(looks_like_a_secret("authorization bearer abcdefghij"));
        assert!(looks_like_a_secret("use sk-abcdefghij for this"));
        // The word alone is not a leak, and neither is a longer name that
        // merely ends in one.
        assert!(!looks_like_a_secret("the token is rotated weekly"));
        assert!(!looks_like_a_secret("my_secret_sauce = onions"));
        assert!(!looks_like_a_secret("sk-short"));
    }

    #[test]
    fn invisible_unicode_is_refused() {
        assert!(reject_unsafe("plain text").is_ok());
        assert!(reject_unsafe("hidden\u{200b}text").is_err());
        assert!(reject_unsafe("\u{feff}bom").is_err());
    }

    #[test]
    fn the_live_set_reads_the_tombstone_and_the_window() {
        let now = "2026-01-01T00:00:00.000Z";
        assert!(is_live(&atom(json!({})), now));
        assert!(is_live(&atom(json!({"valid_to": null})), now));
        assert!(is_live(&atom(json!({"valid_to": ""})), now));
        assert!(is_live(
            &atom(json!({"valid_to": "2099-01-01T00:00:00.000Z"})),
            now
        ));
        assert!(!is_live(
            &atom(json!({"valid_to": "2020-01-01T00:00:00.000Z"})),
            now
        ));
        assert!(!is_live(&atom(json!({"tombstone": true})), now));
    }

    #[test]
    fn the_review_clock_ignores_the_live_window() {
        let now = "2026-01-01T00:00:00.000Z";
        assert!(!is_due(&atom(json!({})), now));
        assert!(!is_due(&atom(json!({"due_at": ""})), now));
        assert!(is_due(
            &atom(json!({"due_at": "2025-01-01T00:00:00.000Z"})),
            now
        ));
        assert!(!is_due(
            &atom(json!({"due_at": "2099-01-01T00:00:00.000Z"})),
            now
        ));
        // Tombstoned and due at once: the two questions do not consult each
        // other, which is why the store asks both.
        let both = atom(json!({"tombstone": true, "due_at": "2025-01-01T00:00:00.000Z"}));
        assert!(!is_live(&both, now));
        assert!(is_due(&both, now));
    }

    #[test]
    fn a_declared_entity_list_wins_over_the_text() {
        let declared = atom(json!({"text": "The Parser reads it.", "entities": ["only-this"]}));
        assert_eq!(
            entities_of(&declared).into_iter().collect::<Vec<_>>(),
            vec!["only-this".to_string()]
        );
        // An empty declared list is still a declaration, so the text is not
        // mined behind the author's back.
        let empty = atom(json!({"text": "The Parser reads it.", "entities": []}));
        assert!(entities_of(&empty).is_empty());
    }

    #[test]
    fn validate_normalizes_the_set_name_and_drops_an_empty_one() {
        let mut a =
            atom(json!({"kind": "voice", "text": "A claim.", "workspace": "w", "set": "Review"}));
        validate(&mut a).unwrap();
        assert_eq!(a["set"], json!("review"));

        let mut b = atom(json!({"kind": "voice", "text": "A claim.", "workspace": "w", "set": ""}));
        validate(&mut b).unwrap();
        assert!(!b.contains_key("set"));
    }

    #[test]
    fn validate_leaves_fields_it_does_not_own_alone() {
        // The store is shared with another writer, so an unmodelled field has
        // to survive the round trip rather than be dropped as unknown.
        let mut a = atom(json!({
            "kind": "voice",
            "text": "A claim.",
            "workspace": "w",
            "something_else": {"nested": [1, 2, 3]}
        }));
        validate(&mut a).unwrap();
        assert_eq!(a["something_else"], json!({"nested": [1, 2, 3]}));
    }

    #[test]
    fn the_text_cap_counts_characters_not_bytes() {
        let wide = "\u{4e00}".repeat(TEXT_SOFT_CAP);
        let mut ok = atom(json!({"kind": "voice", "text": wide, "workspace": "w"}));
        assert!(validate(&mut ok).is_ok());
        let over = "\u{4e00}".repeat(TEXT_SOFT_CAP + 1);
        let mut bad = atom(json!({"kind": "voice", "text": over, "workspace": "w"}));
        assert!(validate(&mut bad).is_err());
    }

    #[test]
    fn scheduling_never_touches_the_live_window() {
        let mut a = atom(json!({"id": "a", "valid_to": "2099-01-01T00:00:00.000Z"}));
        schedule_review(&mut a, "2026-01-01T00:00:00.000Z", Grade::Recalled, None);
        assert_eq!(a["valid_to"], json!("2099-01-01T00:00:00.000Z"));
        assert!(a.contains_key("due_at"));
    }

    #[test]
    fn a_lapse_shortens_and_a_recall_lengthens() {
        let now = "2026-01-01T00:00:00.000Z";
        let block = json!({"reps": 3, "stability": 4.0, "difficulty": 6.0, "last": "2025-12-20T00:00:00.000Z"});
        let mut lapsed = atom(json!({"id": "a", "review": block}));
        let mut recalled = lapsed.clone();
        schedule_review(&mut lapsed, now, Grade::Lapsed, None);
        schedule_review(&mut recalled, now, Grade::Recalled, None);
        let s_lapsed = lapsed["review"]["stability"].as_f64().unwrap();
        let s_recalled = recalled["review"]["stability"].as_f64().unwrap();
        assert!(s_lapsed < 4.0, "{s_lapsed}");
        assert!(s_recalled > 4.0, "{s_recalled}");
        assert_eq!(
            lapsed["review"]["reps"],
            json!(0),
            "a lapse restarts the count"
        );
        assert_eq!(recalled["review"]["reps"], json!(4));
    }
}
