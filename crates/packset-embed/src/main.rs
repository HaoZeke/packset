//! `packset-embed`: the dense projection, driven as a separate process.
//!
//! Same shape as the search projection, and for the same reason. A model and
//! the runtime under it are a native dependency with a download behind them,
//! and most machines that build the writer will never hold one. So this is its
//! own binary, absent is a supported state, and the writer falls back to the
//! scorers it already has.
//!
//! One JSON line in, one JSON line out, flushed, until the input closes.
//! Loading the model is the expensive part, so this is meant to be kept open
//! rather than spawned per question.
//!
//! ```console
//! $ echo '{"id":"a","text":"Prefer ripgrep for search."}' | packset-embed
//! {"id":"a","v":[...]}
//! $ echo '{"id":"q","text":"which search tool"}' | packset-embed --query
//! ```
//!
//! Documents and questions are encoded differently: BGE asks a question to
//! carry a retrieval instruction that a document must not. Passing `--query`
//! for a document, or forgetting it for a question, silently costs recall
//! rather than failing, which is why it is a flag and not a guess. The library
//! applies no prefix of its own, so the instruction is written out here.

use std::io::{BufRead, Write};

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use serde::{Deserialize, Serialize};

/// One thing to encode.
#[derive(Deserialize)]
struct Item {
    id: String,
    text: String,
}

/// One thing encoded.
#[derive(Serialize)]
struct Vector {
    id: String,
    v: Vec<f32>,
}

/// What BGE wants in front of a question, and in front of nothing else.
const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

fn main() -> anyhow::Result<()> {
    let mut query = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--query" | "-q" => query = true,
            "-V" | "--version" => {
                println!("packset-embed {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}\n\n{USAGE}"),
        }
    }

    // Named so a seat can put the weights where its policy allows, and so a
    // build machine and a run machine can share one copy.
    let mut options =
        TextInitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false);
    if let Some(dir) = std::env::var_os("PACKSET_EMBED_CACHE") {
        options = options.with_cache_dir(std::path::PathBuf::from(dir));
    }
    let mut model = TextEmbedding::try_new(options)?;

    // A line in, a line out, flushed. Loading the model is the expensive part
    // and a caller that has to pay it per question cannot afford to ask, so
    // this process is meant to be kept rather than spawned: it answers until
    // its input closes.
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let item: Item = serde_json::from_str(&line)?;
        let text = if query {
            format!("{QUERY_PREFIX}{}", item.text)
        } else {
            item.text
        };
        let mut vectors = model.embed(&[text], None)?;
        let v = vectors.pop().unwrap_or_default();
        writeln!(
            out,
            "{}",
            serde_json::to_string(&Vector { id: item.id, v })?
        )?;
        out.flush()?;
    }
    Ok(())
}

const USAGE: &str = "packset-embed: text in, vectors out\n\
    \n\
    reads JSON lines {\"id\",\"text\"} and writes {\"id\",\"v\"}\n\
    \n\
        --query   encode as a question rather than a document\n\
        PACKSET_EMBED_CACHE   where the weights live";
