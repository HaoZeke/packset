//! Okapi BM25 over the pack, with the postings to answer it.
//!
//! The pack's own scorer answers a different question: it takes the best any
//! query token can do against a text and adds those up, which is what makes a
//! typo and a prefix still find an atom. What it cannot do is weigh a word by
//! how much saying it narrows anything down, or stop a long atom scoring well
//! because it has more words to match with.
//!
//! BM25 does both and nothing else: an exact term, weighted by how rare it is,
//! saturating so a fourth occurrence adds less than the second, and normalised
//! by length against the corpus average. The two scorers disagree about
//! different queries, which is the reason to run both and let the panel fuse
//! them rather than to replace one with the other.
//!
//! The index is here rather than in the caller because a scan is the wrong
//! shape for this scorer. BM25 gives nothing to a document carrying no query
//! term, so touching every document to find that out costs the whole pack per
//! question; the postings cost the answer instead.

use std::collections::HashMap;

/// Term-frequency saturation. The value the literature uses.
const K1: f64 = 1.2;

/// How much length normalisation applies. 0 is none, 1 is full.
const B: f64 = 0.75;

/// Where one term appears: the document, and how often in it.
type Posting = (u32, u32);

/// An inverted index over one corpus, with the statistics BM25 needs.
#[derive(Debug, Clone, Default)]
pub struct Index {
    postings: HashMap<String, Vec<Posting>>,
    lengths: Vec<u32>,
    total_length: u64,
}

impl Index {
    /// Build over a corpus of tokenised documents, in the caller's order.
    ///
    /// The ordinal of a document is its position in that order, so a caller
    /// that keeps the corpus and the index together can go straight from a
    /// score back to what was scored.
    #[must_use]
    pub fn build<'a>(documents: impl IntoIterator<Item = &'a [String]>) -> Self {
        let mut index = Self::default();
        for tokens in documents {
            let ordinal = u32::try_from(index.lengths.len()).unwrap_or(u32::MAX);
            index
                .lengths
                .push(u32::try_from(tokens.len()).unwrap_or(u32::MAX));
            index.total_length += tokens.len() as u64;
            let mut counts: HashMap<&str, u32> = HashMap::new();
            for term in tokens {
                *counts.entry(term.as_str()).or_insert(0) += 1;
            }
            for (term, count) in counts {
                index
                    .postings
                    .entry(term.to_string())
                    .or_default()
                    .push((ordinal, count));
            }
        }
        index
    }

    /// How many documents are indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lengths.len()
    }

    /// Whether anything was indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lengths.is_empty()
    }

    /// Mean document length in tokens.
    #[must_use]
    pub fn average_length(&self) -> f64 {
        if self.lengths.is_empty() {
            0.0
        } else {
            self.total_length as f64 / self.lengths.len() as f64
        }
    }

    /// How much one term narrows the corpus down.
    ///
    /// The `+ 1` inside the logarithm is what keeps this positive for a term in
    /// every document. Without it such a term scores negative, and a document
    /// is then punished for carrying a word everything carries, which is not
    /// what "says nothing" should mean.
    #[must_use]
    pub fn idf(&self, term: &str) -> f64 {
        let n = self.lengths.len() as f64;
        let df = self.postings.get(term).map_or(0, Vec::len) as f64;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// Length normalisation for one document.
    fn norm(&self, ordinal: usize) -> f64 {
        let average = self.average_length();
        if average <= 0.0 {
            return 1.0;
        }
        let length = f64::from(self.lengths.get(ordinal).copied().unwrap_or(0));
        B.mul_add(length / average, 1.0 - B)
    }

    /// One term's contribution to one document.
    fn term_score(&self, term: &str, ordinal: usize, count: u32) -> f64 {
        let count = f64::from(count);
        self.idf(term) * (count * (K1 + 1.0)) / (K1 * self.norm(ordinal)).mul_add(1.0, count)
    }

    /// Every document carrying at least one query term, with its score.
    ///
    /// Ordinals ascend, so the caller's tie-break decides ties rather than a
    /// hash iteration order.
    #[must_use]
    pub fn score(&self, query: &[String]) -> Vec<(usize, f64)> {
        let mut totals: HashMap<u32, f64> = HashMap::new();
        for term in query {
            let Some(postings) = self.postings.get(term.as_str()) else {
                continue;
            };
            for (ordinal, count) in postings {
                *totals.entry(*ordinal).or_insert(0.0) +=
                    self.term_score(term, *ordinal as usize, *count);
            }
        }
        let mut scored: Vec<(usize, f64)> = totals
            .into_iter()
            .map(|(ordinal, score)| (ordinal as usize, score))
            .collect();
        scored.sort_unstable_by_key(|(ordinal, _)| *ordinal);
        scored
    }

    /// Score a document that is not in the index, against this corpus.
    ///
    /// A card paragraph is written the same way an atom is and competes with
    /// one for the same place in an answer, so it is weighed by how rare its
    /// words are among the atoms rather than by a corpus of its own.
    #[must_use]
    pub fn score_foreign(&self, query: &[String], document: &[String]) -> f64 {
        if document.is_empty() || self.is_empty() {
            return 0.0;
        }
        let average = self.average_length();
        let norm = if average > 0.0 {
            B.mul_add(document.len() as f64 / average, 1.0 - B)
        } else {
            1.0
        };
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for term in document {
            *counts.entry(term.as_str()).or_insert(0) += 1;
        }
        query
            .iter()
            .map(|term| {
                let count = f64::from(counts.get(term.as_str()).copied().unwrap_or(0));
                if count == 0.0 {
                    return 0.0;
                }
                self.idf(term) * (count * (K1 + 1.0)) / (K1 * norm).mul_add(1.0, count)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(text: &str) -> Vec<String> {
        crate::search::tokens(text)
    }

    fn corpus(texts: &[&str]) -> Index {
        let docs: Vec<Vec<String>> = texts.iter().map(|t| doc(t)).collect();
        Index::build(docs.iter().map(Vec::as_slice))
    }

    fn best(index: &Index, query: &str) -> usize {
        index
            .score(&doc(query))
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("a hit")
            .0
    }

    /// The whole reason to run this beside the pack's own scorer.
    #[test]
    fn a_rare_word_says_more_than_a_common_one() {
        let index = corpus(&[
            "the parser reads a header",
            "the parser reads a manifest",
            "the parser reads a record",
            "the ripgrep overlay reads a header",
        ]);
        assert!(index.idf("ripgrep") > index.idf("parser"));
        // The atom carrying the rare word wins even though the other three
        // carry a word the query also names.
        assert_eq!(best(&index, "ripgrep parser"), 3);
    }

    /// A term in every document narrows nothing, and must not go negative.
    #[test]
    fn a_word_everything_carries_never_scores_below_nothing() {
        let index = corpus(&["parser one", "parser two", "parser three"]);
        assert!(index.idf("parser") > 0.0);
        let scored = index.score(&doc("parser"));
        assert_eq!(scored.len(), 3);
        assert!(scored.iter().all(|(_, score)| *score > 0.0), "{scored:?}");
    }

    /// Length normalisation: padding an atom must not raise its score.
    #[test]
    fn a_longer_atom_does_not_win_on_length_alone() {
        let padding = "header manifest record token commit branch index atom workspace daemon";
        let index = corpus(&["the parser reads", &format!("the parser reads {padding}")]);
        let scored = index.score(&doc("parser"));
        assert_eq!(scored.len(), 2);
        assert!(scored[0].1 > scored[1].1, "padding raised the score");
    }

    /// Saturation: the second occurrence is worth less than the first.
    #[test]
    fn repeating_a_word_pays_less_each_time() {
        let index = corpus(&["parser", "parser parser", "parser parser parser"]);
        let scored = index.score(&doc("parser"));
        let (once, twice, thrice) = (scored[0].1, scored[1].1, scored[2].1);
        assert!(twice > once);
        assert!(thrice - twice < twice - once);
    }

    /// The point of the postings: a document with no query term is never
    /// touched, let alone returned.
    #[test]
    fn a_document_carrying_no_query_term_is_not_in_the_answer() {
        let index = corpus(&["the parser reads a header", "the overlay writes a record"]);
        assert_eq!(
            index.score(&doc("parser")),
            vec![(0, index.score(&doc("parser"))[0].1)]
        );
        assert!(index.score(&doc("kubernetes")).is_empty());
    }

    #[test]
    fn an_empty_corpus_scores_nothing_rather_than_dividing_by_it() {
        let index = Index::build(std::iter::empty());
        assert!(index.is_empty());
        assert!(index.score(&doc("parser")).is_empty());
        assert_eq!(index.score_foreign(&doc("parser"), &doc("parser")), 0.0);
    }

    /// A card is weighed against the atoms, and the same words score the same
    /// whichever side of the pack they were written on.
    #[test]
    fn a_foreign_document_is_weighed_against_the_indexed_corpus() {
        let index = corpus(&["the parser reads a header", "the parser reads a manifest"]);
        let indexed = index.score(&doc("parser"))[0].1;
        let foreign = index.score_foreign(&doc("parser"), &doc("the parser reads a header"));
        assert!((indexed - foreign).abs() < 1e-9, "{indexed} vs {foreign}");
    }
}
