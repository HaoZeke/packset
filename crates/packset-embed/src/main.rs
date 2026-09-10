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

use fastembed::{
    Bgem3Embedding, Bgem3InitOptions, Bgem3Model, EmbeddingModel, RerankInitOptions, RerankerModel,
    TextEmbedding, TextInitOptions, TextRerank,
};
use serde::{Deserialize, Serialize};

/// One thing to encode.
#[derive(Deserialize)]
struct Item {
    id: String,
    text: String,
}

/// One thing encoded as a single vector.
#[derive(Serialize)]
struct Vector {
    id: String,
    v: Vec<f32>,
}

/// A question and the candidates to put it against.
#[derive(Deserialize)]
struct Pairing {
    id: String,
    /// What was asked.
    q: String,
    /// The candidate texts, in the order the first stage returned them.
    d: Vec<String>,
}

/// What the cross-encoder made of them.
///
/// Scores in the caller's order rather than a reordered list, so the caller
/// keeps the mapping from candidate to atom it already had. Reordering here
/// would hand back a permutation of texts and make the caller match strings
/// back to ids.
#[derive(Serialize)]
struct Scored {
    id: String,
    s: Vec<f32>,
}

/// One thing encoded both ways, from one pass.
///
/// Late interaction scores a document by the best match each query token finds
/// anywhere in it, so it needs the tokens kept apart rather than pooled. That
/// is the whole difference, and it is also the whole cost: a short claim
/// carries thirty vectors where the pooled form carries one.
///
/// The model returns its own pooled vector from the same forward pass, and it
/// rides along so a caller can compare the two scorings with the model held
/// fixed. Comparing this model's tokens against another model's pooling
/// measures both differences at once and attributes them to one.
#[derive(Serialize)]
struct Tokens {
    id: String,
    t: Vec<Vec<f32>>,
    v: Vec<f32>,
    /// The learned term weights from the same pass: which vocabulary entries
    /// this text activates, and how much. One number a term where the token
    /// form is a vector a token, so it costs what an inverted index costs.
    s: Sparse,
}

/// Learned term weights, as the pair of arrays the model returns.
#[derive(Serialize, Default)]
struct Sparse {
    i: Vec<u32>,
    w: Vec<f32>,
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
    let mut late = false;
    let mut rerank = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--query" | "-q" => query = true,
            "--late" => late = true,
            "--rerank" => rerank = true,
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
    if rerank {
        return cross_encode();
    }
    if late {
        return late_interaction(query);
    }

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

/// BGE-M3, which returns a vector per token from the same pass.
///
/// A separate path rather than a model name, because what it emits is a
/// different shape and a caller that asked for one and got the other would
/// score nonsense rather than fail.
fn late_interaction(query: bool) -> anyhow::Result<()> {
    // The int8 quantisation is the only BGE-M3 on offer here, and it is worth
    // saying out loud: a number from it sits a little under what the
    // full-precision model would give, so it bounds late interaction from
    // below rather than measuring it exactly.
    let mut options = Bgem3InitOptions::new(Bgem3Model::BGEM3Q).with_show_download_progress(false);
    if let Some(dir) = std::env::var_os("PACKSET_EMBED_CACHE") {
        options = options.with_cache_dir(std::path::PathBuf::from(dir));
    }
    let mut model = Bgem3Embedding::try_new(options)?;

    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let item: Item = serde_json::from_str(&line)?;
        // BGE-M3 takes no instruction prefix of its own; the flag stays so one
        // caller drives both paths the same way.
        let _ = query;
        let mut encoded = model.embed(&[item.text], None)?;
        let t = if encoded.colbert.is_empty() {
            Vec::new()
        } else {
            encoded.colbert.remove(0)
        };
        // The pooled vector from the same forward pass, so a caller can compare
        // the two scorings without changing the model underneath them.
        let v = if encoded.dense.is_empty() {
            Vec::new()
        } else {
            encoded.dense.remove(0)
        };
        // The learned sparse weights, also from that pass. BGE-M3 is trained to
        // emit all three, and a caller that already paid for the forward pass
        // has this for nothing.
        let s = if encoded.sparse.is_empty() {
            Sparse::default()
        } else {
            let raw = encoded.sparse.remove(0);
            Sparse {
                i: raw.indices.iter().map(|index| *index as u32).collect(),
                w: raw.values,
            }
        };
        writeln!(
            out,
            "{}",
            serde_json::to_string(&Tokens {
                id: item.id,
                t,
                v,
                s
            })?
        )?;
        out.flush()?;
    }
    Ok(())
}

/// The cross-encoder that reads a question and a candidate together.
///
/// Every other path here is a bi-encoder: a text is embedded once, without the
/// question, and the score is a geometry between two vectors made in ignorance
/// of each other. That is what makes a first stage cheap, and it is also its
/// ceiling. A cross-encoder reads the pair in one forward pass, so it can
/// answer whether this text answers this question rather than whether the two
/// are about the same subject.
///
/// The cost is the other half of the trade and it is not small: a bi-encoder
/// embeds a corpus once and answers every question from the stored vectors,
/// where this runs a forward pass per candidate per question. That is why it
/// is a second stage over the top of a ranking rather than a scorer over a
/// pack, and why a seat that turns it on is buying accuracy with latency.
///
/// Nogueira and Cho (doi:10.48550/arXiv.1901.04085) is the result this is;
/// monoT5 (doi:10.18653/v1/2020.findings-emnlp.63) is the same structure with
/// a sequence-to-sequence model.
fn cross_encode() -> anyhow::Result<()> {
    let name = std::env::var("PACKSET_RERANK_MODEL").unwrap_or_default();
    let model = match name.trim() {
        "bge-reranker-base" | "" => RerankerModel::BGERerankerBase,
        "bge-reranker-v2-m3" => RerankerModel::BGERerankerV2M3,
        "jina-turbo" => RerankerModel::JINARerankerV1TurboEn,
        "jina-v2" => RerankerModel::JINARerankerV2BaseMultiligual,
        other => anyhow::bail!("unknown reranker `{other}`; known: {RERANKERS}"),
    };
    let mut options = RerankInitOptions::new(model).with_show_download_progress(false);
    if let Some(dir) = std::env::var_os("PACKSET_EMBED_CACHE") {
        options = options.with_cache_dir(std::path::PathBuf::from(dir));
    }
    let mut reranker = TextRerank::try_new(options)?;

    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let pairing: Pairing = serde_json::from_str(&line)?;
        // An empty candidate list is a question the first stage answered with
        // nothing, which is not an error and must not cost a model call.
        let mut scores = vec![0.0f32; pairing.d.len()];
        if !pairing.d.is_empty() {
            let texts: Vec<&str> = pairing.d.iter().map(String::as_str).collect();
            for result in reranker.rerank(pairing.q.as_str(), &texts, false, None)? {
                if let Some(slot) = scores.get_mut(result.index) {
                    *slot = result.score;
                }
            }
        }
        writeln!(
            out,
            "{}",
            serde_json::to_string(&Scored {
                id: pairing.id,
                s: scores
            })?
        )?;
        out.flush()?;
    }
    Ok(())
}

/// Every reranker [`cross_encode`] answers to, for the error that lists them.
const RERANKERS: &str = "bge-reranker-base, bge-reranker-v2-m3, jina-turbo, jina-v2";

const USAGE: &str = "packset-embed: text in, vectors out\n\
    \n\
    reads JSON lines {\"id\",\"text\"} and writes {\"id\",\"v\"}\n\
    \n\
        --query   encode as a question rather than a document\n\
        --late    a vector per token, for late interaction (BGE-M3)\n\
        --rerank  a cross-encoder second stage: reads {\"id\",\"q\",\"d\":[..]}\n\
                  and writes {\"id\",\"s\":[..]}, one score a candidate in the\n\
                  order given\n\
    \n\
        PACKSET_EMBED_CACHE   where the weights live\n\
        PACKSET_EMBED_MODEL   bge-small (default), bge-base, bge-large,\n\
                              e5-base, e5-large, gte-large, mxbai-large\n\
        PACKSET_RERANK_MODEL  bge-reranker-base (default), bge-reranker-v2-m3,\n\
                              jina-turbo, jina-v2";
