//! The dense projection, driven as a kept process.
//!
//! Same arrangement as the search projection next door, for the same reason. A
//! model and the runtime under it are a native dependency with a download
//! behind them, and most machines that build the writer will never hold one.
//! So the encoder is its own binary, an absent one is a supported state, and
//! every failure here falls back to the scorers the pack already has rather
//! than to a partial answer.
//!
//! Kept rather than spawned, because loading the model costs about a second
//! and a search cannot pay that. One child per direction, since a question and
//! a document are encoded differently and the flag is set at startup.
//!
//! A vector is a projection and never the store: it is derivable from the
//! text, so a missing or stale one costs ranking quality and nothing else.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Mutex, OnceLock};

use serde_json::{json, Value};

/// Environment variables naming the encoder.
pub const BIN_VARS: &[&str] = &["PACKSET_EMBED"];

/// The encoder this seat would run, if it has one.
///
/// The same shape as the search projection's discovery, and deliberately not
/// the workspace target directory: a seat that builds there names the binary
/// with `PACKSET_EMBED`.
#[must_use]
pub fn binary() -> Option<PathBuf> {
    for var in BIN_VARS {
        if let Some(raw) = std::env::var_os(var) {
            let path = PathBuf::from(raw);
            if is_executable(&path) {
                return Some(path);
            }
        }
    }
    let here = std::env::current_exe().ok()?;
    // Beside this binary, which is where a seat that installs the pair puts it.
    if let Some(beside) = here
        .parent()
        .map(|dir| dir.join("packset-embed"))
        .filter(|path| is_executable(path))
    {
        return Some(beside);
    }
    if let Some(root) = here.parent().and_then(Path::parent).and_then(Path::parent) {
        for candidate in [
            root.join("bin/packset-embed"),
            root.join("crates/packset-embed/target/release/packset-embed"),
        ] {
            if is_executable(&candidate) {
                return Some(candidate);
            }
        }
    }
    which("packset-embed")
}

fn which(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join(name))
        .find(|path| is_executable(path))
}

fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
}

/// A running encoder: one child, one line in, one line out.
struct Encoder {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl Encoder {
    fn start(binary: &Path, query: bool) -> Option<Self> {
        let mut command = Command::new(binary);
        if query {
            command.arg("--query");
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let stdin = child.stdin.take()?;
        let stdout = BufReader::new(child.stdout.take()?);
        Some(Self {
            child,
            stdin,
            stdout,
        })
    }

    fn start_rerank(binary: &Path) -> Option<Self> {
        let mut child = Command::new(binary)
            .arg("--rerank")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let stdin = child.stdin.take()?;
        let stdout = BufReader::new(child.stdout.take()?);
        Some(Self {
            child,
            stdin,
            stdout,
        })
    }

    /// One question against many candidates, one score each.
    ///
    /// The whole pairing goes in one line because the model reads the question
    /// and a candidate together; sending a candidate at a time would be a
    /// round trip a candidate for no gain, and the batch is what lets the
    /// child pack them into one forward pass.
    fn rerank(&mut self, question: &str, candidates: &[String]) -> Option<Vec<f32>> {
        let asked = serde_json::json!({ "id": "q", "q": question, "d": candidates });
        let reply = self.ask_json(&asked.to_string())?;
        let parsed: Value = serde_json::from_str(reply.trim()).ok()?;
        let scores: Vec<f32> = parsed
            .get("s")?
            .as_array()?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        // A short answer is a mismatch between what was asked and what came
        // back, and padding it would silently score the tail as zero.
        (scores.len() == candidates.len()).then_some(scores)
    }

    fn start_sparse(binary: &Path) -> Option<Self> {
        let mut child = Command::new(binary)
            .arg("--sparse")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let stdin = child.stdin.take()?;
        let stdout = BufReader::new(child.stdout.take()?);
        Some(Self {
            child,
            stdin,
            stdout,
        })
    }

    /// One line in, the learned term weights out.
    fn encode_sparse(&mut self, text: &str) -> Option<Sparse> {
        let reply = self.ask(text)?;
        let parsed: Value = serde_json::from_str(reply.trim()).ok()?;
        let s = parsed.get("s")?;
        let indices = s.get("i")?.as_array()?;
        let weights = s.get("w")?.as_array()?;
        if indices.len() != weights.len() {
            return None;
        }
        let mut pairs: Sparse = indices
            .iter()
            .zip(weights)
            .filter_map(|(i, w)| Some((i.as_u64()? as u32, w.as_f64()? as f32)))
            .collect();
        // Ascending by index, which is what lets two of them intersect in one
        // pass; the model does not promise an order.
        pairs.sort_unstable_by_key(|(index, _)| *index);
        Some(pairs)
    }

    fn start_late(binary: &Path) -> Option<Self> {
        let mut child = Command::new(binary)
            .arg("--late")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let stdin = child.stdin.take()?;
        let stdout = BufReader::new(child.stdout.take()?);
        Some(Self {
            child,
            stdin,
            stdout,
        })
    }

    /// One line in, one line carrying all three forms out.
    fn encode_tokens(&mut self, text: &str) -> Option<(Vec<Vec<f32>>, Vec<f32>, Sparse)> {
        let reply = self.ask(text)?;
        let parsed: Value = serde_json::from_str(reply.trim()).ok()?;
        let rows = parsed.get("t")?.as_array()?;
        let tokens: Vec<Vec<f32>> = rows
            .iter()
            .filter_map(|row| {
                let vector: Vec<f32> = row
                    .as_array()?
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                (!vector.is_empty()).then_some(vector)
            })
            .collect();
        let pooled: Vec<f32> = parsed
            .get("v")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();
        let sparse = parsed
            .get("s")
            .map(|raw| {
                let indices = raw
                    .get("i")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|value| value.as_u64().map(|index| index as u32));
                let weights = raw
                    .get("w")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|value| value.as_f64().map(|weight| weight as f32));
                indices.zip(weights).collect()
            })
            .unwrap_or_default();
        let mut sparse: Sparse = sparse;
        sparse.sort_unstable_by_key(|(index, _)| *index);
        (!tokens.is_empty()).then_some((tokens, pooled, sparse))
    }

    /// Write one request and read its one-line reply.
    fn ask(&mut self, text: &str) -> Option<String> {
        let line = json!({ "id": "0", "text": text });
        self.ask_json(&line.to_string())
    }

    /// One line written, one line read back.
    fn ask_json(&mut self, line: &str) -> Option<String> {
        writeln!(self.stdin, "{line}").ok()?;
        self.stdin.flush().ok()?;
        let mut reply = String::new();
        if self.stdout.read_line(&mut reply).ok()? == 0 {
            return None;
        }
        Some(reply)
    }

    fn encode(&mut self, text: &str) -> Option<Vec<f32>> {
        let reply = self.ask(text)?;
        let parsed: Value = serde_json::from_str(reply.trim()).ok()?;
        let vector: Vec<f32> = parsed
            .get("v")?
            .as_array()?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        (!vector.is_empty()).then_some(vector)
    }

    /// Whether the child is still there to be asked.
    fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

/// Learned term weights: which vocabulary entries a text activates, and how
/// much. Ascending by index, so two of them intersect in one pass.
pub type Sparse = Vec<(u32, f32)>;

/// The two kept encoders, started on first use.
type Slot = Mutex<Option<Encoder>>;

fn slot(query: bool) -> &'static Slot {
    static DOCUMENTS: OnceLock<Slot> = OnceLock::new();
    static QUESTIONS: OnceLock<Slot> = OnceLock::new();
    if query {
        QUESTIONS.get_or_init(|| Mutex::new(None))
    } else {
        DOCUMENTS.get_or_init(|| Mutex::new(None))
    }
}

/// Encode one text, or nothing when this seat has no working encoder.
///
/// A child that has died is replaced once and the text retried, because the
/// common way for one to die is the machine reclaiming memory rather than
/// anything about the request. A second failure is reported as no encoder,
/// which is a state every caller already handles.
#[must_use]
pub fn encode(text: &str, query: bool) -> Option<Vec<f32>> {
    if text.trim().is_empty() {
        return None;
    }
    let binary = binary()?;
    let mut held = slot(query).lock().ok()?;
    for attempt in 0..2 {
        if held.as_mut().is_none_or(|running| !running.alive()) {
            *held = Encoder::start(&binary, query);
        }
        let running = held.as_mut()?;
        if let Some(vector) = running.encode(text) {
            return Some(vector);
        }
        *held = None;
        if attempt == 1 {
            return None;
        }
    }
    None
}

/// Encode one query.
#[must_use]
pub fn encode_query(text: &str) -> Option<Vec<f32>> {
    encode(text, true)
}

/// Encode one atom's text.
#[must_use]
pub fn encode_document(text: &str) -> Option<Vec<f32>> {
    encode(text, false)
}

/// The kept encoder for the per-token form, which is a third child.
fn late_slot() -> &'static Slot {
    static LATE: OnceLock<Slot> = OnceLock::new();
    LATE.get_or_init(|| Mutex::new(None))
}

/// Encode one text three ways from one pass: a vector per token, the model's
/// own pooled vector, and its learned term weights.
///
/// A different binary mode rather than a model name, because the shape it
/// returns is different: a caller that asked for one and got the other would
/// score nonsense rather than fail. All three come back so a caller can compare
/// the scorings with the model held fixed, and because the pass has already
/// been paid for by the time any one of them is wanted. Nothing in the writer
/// reads this; it exists so the retrieval benchmark can ask which of the three
/// is the gap.
#[must_use]
pub fn encode_late(text: &str) -> Option<(Vec<Vec<f32>>, Vec<f32>, Sparse)> {
    if text.trim().is_empty() {
        return None;
    }
    let binary = binary()?;
    let mut held = late_slot().lock().ok()?;
    for attempt in 0..2 {
        if held.as_mut().is_none_or(|running| !running.alive()) {
            *held = Encoder::start_late(&binary);
        }
        let running = held.as_mut()?;
        if let Some(both) = running.encode_tokens(text) {
            return Some(both);
        }
        *held = None;
        if attempt == 1 {
            return None;
        }
    }
    None
}

/// The kept learned-sparse encoder, a fifth child.
fn sparse_slot() -> &'static Slot {
    static SPARSE: OnceLock<Slot> = OnceLock::new();
    SPARSE.get_or_init(|| Mutex::new(None))
}

/// Learned term weights from a model trained to produce them.
///
/// BGE-M3's sparse head, which [`encode_late`] also returns, is a side output
/// of a dense model and measured below BM25. SPLADE
/// (doi:10.1145/3404835.3463098) is trained for the weights, with a
/// regularizer that keeps them sparse enough for an inverted index. The two
/// are both "learned sparse" and they are not the same measurement. Nothing in
/// the writer reads this; it exists so the benchmark can ask the right one.
#[must_use]
pub fn encode_sparse(text: &str) -> Option<Sparse> {
    if text.trim().is_empty() {
        return None;
    }
    let binary = binary()?;
    let mut held = sparse_slot().lock().ok()?;
    for attempt in 0..2 {
        if held.as_mut().is_none_or(|running| !running.alive()) {
            *held = Encoder::start_sparse(&binary);
        }
        let running = held.as_mut()?;
        if let Some(weights) = running.encode_sparse(text) {
            return Some(weights);
        }
        *held = None;
        if attempt == 1 {
            return None;
        }
    }
    None
}

/// How deep the second stage reads.
///
/// Reranking the whole first-stage ranking would cost the whole ranking, and
/// the point of a two-stage system is that the second one reads few. Twenty is
/// the deepest cut-off the locomo table scores, so every rank that table
/// reports is inside the window and nothing below it can be promoted into view.
pub const RERANK_DEPTH: usize = 20;

/// Whether the live search path should run the second stage.
///
/// Off unless asked. The first stage embeds a corpus once and answers every
/// question from what it stored; a cross-encoder runs a forward pass per
/// candidate per question. The same three spellings `/v1/search?rerank=`
/// accepts, so a host env and a query flag cannot disagree about what "on" is.
#[must_use]
pub fn wanted() -> bool {
    flag_on(&std::env::var("PACKSET_RERANK").unwrap_or_default())
}

/// Whether one request should run the stage.
///
/// A query flag overrides the host default so a seat can spend the cost on
/// one question without restarting. The same three spellings as the other
/// `/v1` flags; anything else, including empty, is off.
#[must_use]
pub fn requested(query: Option<&str>) -> bool {
    match query.map(str::trim).filter(|s| !s.is_empty()) {
        Some(raw) => flag_on(raw),
        None => wanted(),
    }
}

fn flag_on(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes"
    )
}

/// Reorder the head of a ranking by cross-encoder scores.
///
/// `scores` must cover the head (`RERANK_DEPTH.min(hits.len())`). A length
/// mismatch is refused rather than padded, the same way a short child reply is.
/// The tail keeps its first-stage order. Ties are stable so the first stage
/// breaks them rather than a sort order nobody chose.
#[must_use]
pub fn apply_rerank(hits: &[Value], scores: &[f32]) -> Option<Vec<Value>> {
    let depth = RERANK_DEPTH.min(hits.len());
    if scores.len() != depth {
        return None;
    }
    let mut head: Vec<(f32, Value)> = scores
        .iter()
        .copied()
        .zip(hits[..depth].iter().cloned())
        .collect();
    head.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out: Vec<Value> = head.into_iter().map(|(_, hit)| hit).collect();
    out.extend_from_slice(&hits[depth..]);
    Some(out)
}

/// Reorder the top of a ranking by what a cross-encoder makes of it.
///
/// The same stage the locomo arm measures and `/v1/search` runs when asked.
/// An absent or broken reranker is `None` so the caller can leave the ranking
/// as it was and say so, rather than silently reordering by a stage that did
/// not run.
#[must_use]
pub fn rerank_hits(question: &str, hits: &[Value]) -> Option<Vec<Value>> {
    if hits.is_empty() {
        return Some(Vec::new());
    }
    let depth = RERANK_DEPTH.min(hits.len());
    let candidates: Vec<String> = hits[..depth]
        .iter()
        .map(|hit| {
            hit.get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    let scores = rerank(question, &candidates)?;
    apply_rerank(hits, &scores)
}

/// The kept cross-encoder, a fourth child.
fn rerank_slot() -> &'static Slot {
    static RERANK: OnceLock<Slot> = OnceLock::new();
    RERANK.get_or_init(|| Mutex::new(None))
}

/// Score every candidate against the question, reading the pair together.
///
/// Not the panel's `rerank`, which is maximal marginal relevance over a
/// ranking the panel already has and answers "which of these are redundant".
/// This one asks a model "does this answer the question", which is the thing
/// nothing in the first stage can be asked.
///
/// This is the second stage the first-stage scorers cannot be. Every other
/// path here embeds a text without the question, so what it compares is two
/// vectors made in ignorance of each other; a cross-encoder reads the pair in
/// one forward pass and can answer whether this text answers this question
/// rather than whether the two are about the same subject
/// (doi:10.48550/arXiv.1901.04085).
///
/// The cost is the reason it is a stage and not a scorer. A bi-encoder embeds
/// a corpus once and answers every question from the stored vectors; this runs
/// a forward pass per candidate per question, so it is run over the top of a
/// ranking that has already thrown most of the corpus away.
///
/// Scores come back in the caller's order, unsorted, because what the panel
/// wants is a ballot rather than a decision.
#[must_use]
pub fn rerank(question: &str, candidates: &[String]) -> Option<Vec<f32>> {
    if question.trim().is_empty() {
        return None;
    }
    // Nothing to score is not a failure, and it must not cost a model call.
    if candidates.is_empty() {
        return Some(Vec::new());
    }
    let binary = binary()?;
    let mut held = rerank_slot().lock().ok()?;
    for attempt in 0..2 {
        if held.as_mut().is_none_or(|running| !running.alive()) {
            *held = Encoder::start_rerank(&binary);
        }
        let running = held.as_mut()?;
        if let Some(scores) = running.rerank(question, candidates) {
            return Some(scores);
        }
        *held = None;
        if attempt == 1 {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_that_is_not_a_program_is_not_an_encoder() {
        assert!(!is_executable(&PathBuf::from("/nonexistent/packset-embed")));
        assert!(!is_executable(&PathBuf::from("/etc")));
    }

    #[test]
    fn empty_text_is_never_sent_to_a_model() {
        assert!(encode("", false).is_none());
        assert!(encode("   ", true).is_none());
    }

    /// A question the first stage answered with nothing costs no model call
    /// and is not an error.
    ///
    /// The two are different answers and the caller acts on them differently:
    /// an empty ballot leaves the ranking alone, and `None` means there is no
    /// reranker, which the arm reports rather than silently reordering by a
    /// stage that did not run.
    #[test]
    fn nothing_to_rerank_is_an_empty_ballot_and_not_a_failure() {
        assert_eq!(rerank("which search tool", &[]), Some(Vec::new()));
        // An empty question is refused before any child is started, the same
        // way an empty text is never sent to an encoder.
        assert!(rerank("", &["a candidate".to_string()]).is_none());
        assert!(rerank("   ", &["a candidate".to_string()]).is_none());
    }

    /// A reply that scores fewer candidates than were asked about is refused.
    ///
    /// Padding it would score the tail as zero, which reads as a candidate the
    /// model rejected rather than one it never saw. The two are indistinguishable
    /// downstream, which is what makes the short reply worth refusing here.
    #[test]
    fn a_short_reply_is_a_mismatch_rather_than_a_ranking() {
        let Ok(mut child) = Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
        else {
            return;
        };
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            return;
        };
        // `cat` echoes what it is given, so the reply carries the request's
        // own fields and no `s` at all: a well-formed line that is not an
        // answer.
        let mut echoing = Encoder {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let candidates = vec!["one".to_string(), "two".to_string()];
        assert!(echoing.rerank("a question", &candidates).is_none());
    }

    #[test]
    fn a_child_that_exits_is_not_alive() {
        let Ok(mut child) = Command::new("true").stdout(Stdio::piped()).spawn() else {
            return;
        };
        let _ = child.wait();
        assert!(matches!(child.try_wait(), Ok(Some(_))));
    }

    fn hit(id: &str, text: &str) -> Value {
        json!({ "id": id, "text": text })
    }

    /// The second stage is off unless the host asks. An unset env is the
    /// shipped default; a process that already exported PACKSET_RERANK is a
    /// different seat and this test does not speak for it.
    #[test]
    fn the_second_stage_is_off_unless_asked() {
        if std::env::var_os("PACKSET_RERANK").is_some() {
            return;
        }
        assert!(!wanted());
        assert!(!requested(None));
        assert!(!requested(Some("")));
        assert!(!requested(Some("0")));
        assert!(!requested(Some("on")));
        assert!(requested(Some("1")));
        assert!(requested(Some("true")));
        assert!(requested(Some("yes")));
    }

    #[test]
    fn apply_rerank_promotes_the_higher_score_and_keeps_the_tail() {
        let hits: Vec<Value> = (0..22)
            .map(|i| hit(&format!("h{i}"), &format!("text {i}")))
            .collect();
        let mut scores = vec![0.0f32; RERANK_DEPTH];
        scores[0] = 0.1;
        scores[1] = 0.9;
        let ranked = apply_rerank(&hits, &scores).expect("length matches");
        assert_eq!(ranked[0]["id"], json!("h1"));
        assert_eq!(ranked[1]["id"], json!("h0"));
        assert_eq!(ranked[2]["id"], json!("h2"));
        assert_eq!(ranked[20]["id"], json!("h20"));
        assert_eq!(ranked[21]["id"], json!("h21"));
        assert_eq!(ranked.len(), 22);
    }

    #[test]
    fn a_tie_keeps_the_first_stage_order() {
        let hits = vec![hit("first", "a"), hit("second", "b")];
        let ranked = apply_rerank(&hits, &[0.5, 0.5]).expect("length matches");
        assert_eq!(ranked[0]["id"], json!("first"));
        assert_eq!(ranked[1]["id"], json!("second"));
    }

    #[test]
    fn a_short_score_list_is_refused_rather_than_padded() {
        let hits = vec![hit("a", "a"), hit("b", "b")];
        assert!(apply_rerank(&hits, &[0.9]).is_none());
    }

    #[test]
    fn nothing_to_reorder_is_an_empty_ranking() {
        assert_eq!(rerank_hits("which search tool", &[]), Some(Vec::new()));
    }

    /// A stub child that scores later candidates higher, then apply_rerank.
    ///
    /// This is the protocol `/v1/search` and the locomo arm share, without
    /// loading a model.
    #[test]
    fn a_child_that_scores_the_tail_first_reorders_the_head() {
        let script =
            std::env::temp_dir().join(format!("packset-rerank-stub-{}.py", std::process::id()));
        let body = concat!(
            "#!/usr/bin/env python3\n",
            "import json, sys\n",
            "for line in sys.stdin:\n",
            "    line = line.strip()\n",
            "    if not line:\n",
            "        continue\n",
            "    req = json.loads(line)\n",
            "    d = req.get('d', [])\n",
            "    print(json.dumps({'id': req.get('id', 'q'), 's': list(range(len(d)))}), flush=True)\n",
        );
        if std::fs::write(&script, body).is_err() {
            return;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755));
        }
        let Ok(mut child) = Command::new(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
        else {
            let _ = std::fs::remove_file(&script);
            return;
        };
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            let _ = std::fs::remove_file(&script);
            return;
        };
        let mut enc = Encoder {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let candidates = vec!["first".into(), "second".into()];
        let scores = enc.rerank("a question", &candidates);
        drop(enc);
        let _ = std::fs::remove_file(&script);
        let scores = scores.expect("stub scored");
        let hits = vec![hit("a", "first"), hit("b", "second")];
        let ranked = apply_rerank(&hits, &scores).expect("length matches");
        assert_eq!(ranked[0]["id"], json!("b"));
        assert_eq!(ranked[1]["id"], json!("a"));
    }
}
