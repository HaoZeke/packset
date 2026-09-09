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
//! Documents and questions are encoded differently, and differently again per
//! family: BGE asks a question to carry a retrieval instruction that a document
//! must not, and E5 wants a word on both sides. Passing `--query` for a
//! document, or the wrong pair for a model, silently costs recall rather than
//! failing. The library applies no prefix of its own, so both live here beside
//! the name they belong to.

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

/// A model, and the instructions its family wants in front of a text.
///
/// The prefixes are part of the model rather than decoration. BGE was trained
/// with a retrieval instruction on the question only; E5 was trained with a
/// word on both sides. Using the wrong pair costs recall silently, which is
/// why they live beside the name they belong to instead of being a default.
struct Choice {
    model: EmbeddingModel,
    query: &'static str,
    passage: &'static str,
}

/// BGE's instruction, on the question only.
const BGE_QUERY: &str = "Represent this sentence for searching relevant passages: ";

/// The models this binary will load, by the name a seat writes.
fn choose(name: &str) -> Option<Choice> {
    let (model, query, passage) = match name {
        "bge-small" | "" => (EmbeddingModel::BGESmallENV15, BGE_QUERY, ""),
        "bge-base" => (EmbeddingModel::BGEBaseENV15, BGE_QUERY, ""),
        "bge-large" => (EmbeddingModel::BGELargeENV15, BGE_QUERY, ""),
        "e5-large" => (EmbeddingModel::MultilingualE5Large, "query: ", "passage: "),
        "e5-base" => (EmbeddingModel::MultilingualE5Base, "query: ", "passage: "),
        "gte-large" => (EmbeddingModel::GTELargeENV15, "", ""),
        "mxbai-large" => (
            EmbeddingModel::MxbaiEmbedLargeV1,
            "Represent this sentence for searching relevant passages: ",
            "",
        ),
        _ => return None,
    };
    Some(Choice {
        model,
        query,
        passage,
    })
}

/// Every name [`choose`] answers to, for the error that lists them.
const KNOWN: &str = "bge-small, bge-base, bge-large, e5-base, e5-large, gte-large, mxbai-large";

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
    let name = std::env::var("PACKSET_EMBED_MODEL").unwrap_or_default();
    let choice = choose(name.trim())
        .ok_or_else(|| anyhow::anyhow!("unknown model `{name}`; known: {KNOWN}"))?;
    let prefix = if query { choice.query } else { choice.passage };
    let mut options = TextInitOptions::new(choice.model).with_show_download_progress(false);
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
        let text = if prefix.is_empty() {
            item.text
        } else {
            format!("{prefix}{}", item.text)
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
    \n\
        PACKSET_EMBED_CACHE   where the weights live\n\
        PACKSET_EMBED_MODEL   bge-small (default), bge-base, bge-large,\n\
                              e5-base, e5-large, gte-large, mxbai-large";
