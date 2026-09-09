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

    #[test]
    fn a_child_that_exits_is_not_alive() {
        let Ok(mut child) = Command::new("true").stdout(Stdio::piped()).spawn() else {
            return;
        };
        let _ = child.wait();
        assert!(matches!(child.try_wait(), Ok(Some(_))));
    }
}
