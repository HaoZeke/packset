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

/// Lower bound on a term's contribution (BM25+).
///
/// Length normalisation drives one occurrence in a long document toward zero,
/// the value of an absence. The floor keeps an occurrence worth a fixed amount
/// more than none, whatever the length. Lv and Zhai, doi:10.1145/2063576.2063584.
const DELTA: f64 = 1.0;

/// Dirichlet prior for query likelihood, in pseudo-tokens from the collection.
///
/// A document shorter than this is smoothed toward the collection, a longer
/// one toward itself; that is the whole length behaviour. Zhai and Lafferty,
/// doi:10.1145/984321.984322. Taken from the paper, not fitted here.
const MU: f64 = 2000.0;

/// Which scoring family answers a query.
///
/// Three derivations, not three settings: BM25, BM25 with a floor under an
/// occurrence, and query likelihood under a Dirichlet prior. They disagree,
/// which is what a fusion panel is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scorer {
    /// Okapi BM25, with no floor.
    Bm25,
    /// BM25 with a floor under each occurrence. The default; it leads plain
    /// BM25 at every granularity measured, see the README.
    #[default]
    Bm25Plus,
    /// Query likelihood with a Dirichlet prior.
    Dirichlet,
}

impl Scorer {
    /// The name this scorer is asked for by, on a command line or in an
    /// environment variable.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Self::Bm25 => "bm25",
            Self::Bm25Plus => "bm25+",
            Self::Dirichlet => "dirichlet",
        }
    }

    /// Read one from a name, or nothing when the name is not a scorer.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "bm25" | "okapi" => Some(Self::Bm25),
            "bm25+" | "bm25plus" => Some(Self::Bm25Plus),
            "dirichlet" | "ql" | "lm" => Some(Self::Dirichlet),
            _ => None,
        }
    }
}

/// Where one term appears: the document, and how often in it.
type Posting = (u32, u32);

/// An inverted index over one corpus, with the statistics BM25 needs.
#[derive(Debug, Clone, Default)]
pub struct Index {
    postings: HashMap<String, Vec<Posting>>,
    lengths: Vec<u32>,
    total_length: u64,
    /// Collection frequency per term, counting repeats: what a language model
    /// smooths toward, where document frequency says how much a term narrows.
    occurrences: HashMap<String, u64>,
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
                *index.occurrences.entry(term.to_string()).or_insert(0) += u64::from(count);
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

    /// How likely the corpus was to say a term, over all its occurrences.
    fn background(&self, term: &str) -> f64 {
        if self.total_length == 0 {
            return 0.0;
        }
        self.occurrences.get(term).copied().unwrap_or(0) as f64 / self.total_length as f64
    }

    /// One term's contribution to one document.
    fn term_score(&self, term: &str, ordinal: usize, count: u32) -> f64 {
        self.term_score_by(Scorer::Bm25, term, ordinal, count)
    }

    /// One term's contribution, in the family the caller named.
    fn term_score_by(&self, scorer: Scorer, term: &str, ordinal: usize, count: u32) -> f64 {
        let count = f64::from(count);
        match scorer {
            Scorer::Bm25 => {
                self.idf(term) * (count * (K1 + 1.0))
                    / (K1 * self.norm(ordinal)).mul_add(1.0, count)
            }
            // The floor is inside the idf weight: a term that narrows nothing
            // earns nothing for appearing.
            Scorer::Bm25Plus => {
                self.idf(term)
                    * ((count * (K1 + 1.0)) / (K1 * self.norm(ordinal)).mul_add(1.0, count) + DELTA)
            }
            // Per matching term, as Lucene and Anserini do, so the postings
            // can answer it.
            Scorer::Dirichlet => {
                let background = self.background(term);
                if background <= 0.0 {
                    return 0.0;
                }
                let length = f64::from(self.lengths.get(ordinal).copied().unwrap_or(0));
                (count / (MU * background)).ln_1p() + (MU / (length + MU)).ln()
            }
        }
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

    /// Score against a query whose terms carry weights.
    ///
    /// The unweighted form is this with every weight one, which is what a
    /// query somebody typed is: each word asked for once, none of them worth
    /// more than another. An expansion has to say how much less its guesses
    /// count than the words actually asked for, so it needs the weights.
    #[must_use]
    pub fn score_weighted(&self, query: &[(String, f64)]) -> Vec<(usize, f64)> {
        self.score_weighted_by(Scorer::default(), query)
    }

    /// Score a weighted query in the family the caller named. The index is the
    /// same for all three; the formula is chosen per query.
    #[must_use]
    pub fn score_weighted_by(&self, scorer: Scorer, query: &[(String, f64)]) -> Vec<(usize, f64)> {
        let mut totals: HashMap<u32, f64> = HashMap::new();
        for (term, weight) in query {
            if *weight <= 0.0 {
                continue;
            }
            let Some(postings) = self.postings.get(term.as_str()) else {
                continue;
            };
            for (ordinal, count) in postings {
                *totals.entry(*ordinal).or_insert(0.0) +=
                    weight * self.term_score_by(scorer, term, *ordinal as usize, *count);
            }
        }
        let mut scored: Vec<(usize, f64)> = totals
            .into_iter()
            .map(|(ordinal, score)| (ordinal as usize, score))
            .collect();
        scored.sort_unstable_by_key(|(ordinal, _)| *ordinal);
        scored
    }

    /// A query expanded from the documents its own first pass returned.
    ///
    /// The question a person asks names a few of the words the answer uses. A
    /// relevance model estimates the rest from what came back: terms frequent
    /// in the top documents and rare in the corpus are what the asker meant and
    /// did not say. Nothing is trained and nothing is stored, so a pack pays
    /// one extra scoring pass and no model.
    ///
    /// `alpha` is how much of the final query stays the words asked for. The
    /// original terms keep that share because feedback taken on trust is how
    /// this drifts: the first pass is not a relevance judgement, and a wrong
    /// top document expands into more of itself.
    ///
    /// Lavrenko, Croft, Relevance based language models, SIGIR 2001,
    /// doi:10.1145/383952.383972. The interpolation with the original query is
    /// RM3; Lv, Zhai, doi:10.1145/1645953.1646259 for why it is the estimate
    /// worth using.
    #[must_use]
    pub fn expand(
        &self,
        query: &[String],
        feedback: &[(&[String], f64)],
        terms: usize,
        alpha: f64,
    ) -> Vec<(String, f64)> {
        let mut weights: HashMap<String, f64> = HashMap::new();
        // The words asked for, each worth its share of `alpha`.
        if !query.is_empty() {
            let each = alpha / query.len() as f64;
            for term in query {
                *weights.entry(term.clone()).or_insert(0.0) += each;
            }
        }
        let mass: f64 = feedback.iter().map(|(_, score)| score.max(0.0)).sum();
        if mass <= 0.0 || terms == 0 || alpha >= 1.0 {
            let mut out: Vec<(String, f64)> = weights.into_iter().collect();
            out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            return out;
        }
        // P(t | R), summed over the feedback documents by how well each scored.
        let mut model: HashMap<&str, f64> = HashMap::new();
        for (tokens, score) in feedback {
            if tokens.is_empty() || *score <= 0.0 {
                continue;
            }
            let share = score / mass;
            let length = tokens.len() as f64;
            let mut counts: HashMap<&str, u32> = HashMap::new();
            for term in *tokens {
                *counts.entry(term.as_str()).or_insert(0) += 1;
            }
            for (term, count) in counts {
                *model.entry(term).or_insert(0.0) += share * f64::from(count) / length;
            }
        }
        // P(t | R) as it stands. Weighting it by rarity here would apply the
        // rarity twice, since BM25 weights every term by its own idf when it
        // scores: a rare word in one short feedback document would arrive
        // carrying idf squared, which is how an expansion ends up ranked by
        // whichever name happened to appear once.
        let mut ranked: Vec<(&str, f64)> = model.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        ranked.truncate(terms);
        let total: f64 = ranked.iter().map(|(_, value)| *value).sum();
        if total > 0.0 {
            for (term, value) in ranked {
                *weights.entry(term.to_string()).or_insert(0.0) += (1.0 - alpha) * value / total;
            }
        }
        let mut out: Vec<(String, f64)> = weights.into_iter().collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
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

    /// The same, against a query whose terms carry weights.
    #[must_use]
    pub fn score_foreign_weighted(&self, query: &[(String, f64)], document: &[String]) -> f64 {
        self.score_foreign_weighted_by(Scorer::default(), query, document)
    }

    /// A text outside the corpus, scored in the family the caller named, so a
    /// card ranked beside atoms is scored the way they are.
    #[must_use]
    pub fn score_foreign_weighted_by(
        &self,
        scorer: Scorer,
        query: &[(String, f64)],
        document: &[String],
    ) -> f64 {
        if document.is_empty() || self.is_empty() {
            return 0.0;
        }
        let average = self.average_length();
        let length = document.len() as f64;
        let norm = if average > 0.0 {
            B.mul_add(length / average, 1.0 - B)
        } else {
            1.0
        };
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for term in document {
            *counts.entry(term.as_str()).or_insert(0) += 1;
        }
        query
            .iter()
            .map(|(term, weight)| {
                let count = f64::from(counts.get(term.as_str()).copied().unwrap_or(0));
                if count == 0.0 || *weight <= 0.0 {
                    return 0.0;
                }
                let saturated = (count * (K1 + 1.0)) / (K1 * norm).mul_add(1.0, count);
                weight
                    * match scorer {
                        Scorer::Bm25 => self.idf(term) * saturated,
                        Scorer::Bm25Plus => self.idf(term) * (saturated + DELTA),
                        Scorer::Dirichlet => {
                            let background = self.background(term);
                            if background <= 0.0 {
                                return 0.0;
                            }
                            (count / (MU * background)).ln_1p() + (MU / (length + MU)).ln()
                        }
                    }
            })
            .sum()
    }
}

#[cfg(test)]
mod scorers {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_string).collect()
    }

    /// One long document carrying the term among many short ones without it:
    /// BM25 leaves the occurrence worth almost nothing, BM25+ does not.
    #[test]
    fn a_long_document_stops_being_punished_for_its_length() {
        let filler = "alpha beta gamma delta epsilon zeta eta theta ".repeat(400);
        let long = words(&format!("{filler} lease"));
        let mut corpus: Vec<Vec<String>> = vec![long];
        // Enough short documents that the average length is short and the long
        // one is far above it, which is where the normalisation bites.
        for n in 0..200 {
            corpus.push(words(&format!("alpha beta gamma note {n}")));
        }
        let index = Index::build(corpus.iter().map(Vec::as_slice));

        let query = vec![("lease".to_string(), 1.0)];
        let plain = index.score_weighted_by(Scorer::Bm25, &query);
        let floored = index.score_weighted_by(Scorer::Bm25Plus, &query);

        // Only the long document carries the term at all, so both find it.
        assert_eq!(plain.len(), 1);
        assert_eq!(floored.len(), 1);

        let (_, thin) = plain[0];
        let (_, held) = floored[0];
        assert!(held > thin, "the floor took a point away: {held} vs {thin}");
        assert!(
            thin < 0.25 * index.idf("lease"),
            "this corpus does not show the defect: {thin}"
        );
        assert!(
            held > index.idf("lease"),
            "the floor did not restore the occurrence: {held}"
        );
    }

    /// Query likelihood agrees with BM25 on the best document and disagrees on
    /// the numbers, which is the reason to fuse rather than pick.
    #[test]
    fn the_language_model_is_a_different_opinion() {
        let corpus: Vec<Vec<String>> = vec![
            words("lease lease lease renew renew"),
            words("lease renew claim generation fence token holder quiet reclaim node"),
            words("claim generation fence token"),
        ];
        let index = Index::build(corpus.iter().map(Vec::as_slice));
        let query = vec![("lease".to_string(), 1.0), ("renew".to_string(), 1.0)];

        let best = |scored: Vec<(usize, f64)>| -> usize {
            scored
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"))
                .expect("a hit")
                .0
        };
        assert_eq!(best(index.score_weighted_by(Scorer::Bm25, &query)), 0);
        assert_eq!(best(index.score_weighted_by(Scorer::Dirichlet, &query)), 0);

        let by_bm25 = index.score_weighted_by(Scorer::Bm25, &query);
        let by_lm = index.score_weighted_by(Scorer::Dirichlet, &query);
        assert_eq!(by_bm25.len(), by_lm.len());
        assert!(
            by_bm25
                .iter()
                .zip(&by_lm)
                .any(|((_, one), (_, two))| (one - two).abs() > 1e-9),
            "two derivations returned the same numbers"
        );
    }

    /// An unknown scorer name is refused, not defaulted.
    #[test]
    fn a_scorer_is_named_or_refused() {
        for (name, want) in [
            ("bm25", Scorer::Bm25),
            ("BM25+", Scorer::Bm25Plus),
            (" dirichlet ", Scorer::Dirichlet),
            ("ql", Scorer::Dirichlet),
        ] {
            assert_eq!(Scorer::parse(name), Some(want), "{name}");
        }
        assert_eq!(Scorer::parse("tf-idf"), None);
        assert_eq!(Scorer::parse(""), None);
        for scorer in [Scorer::Bm25, Scorer::Bm25Plus, Scorer::Dirichlet] {
            assert_eq!(Scorer::parse(scorer.token()), Some(scorer));
        }
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

    /// The point of the expansion: a word the asker did not say, taken from
    /// what the first pass returned, reaches a document the question misses.
    #[test]
    fn an_expansion_reaches_what_the_question_did_not_say() {
        let index = corpus(&[
            "the parser reads a ripgrep header",
            "the ripgrep overlay writes a header",
            "the kubernetes operator reconciles a deployment",
        ]);
        let documents: Vec<Vec<String>> = [
            "the parser reads a ripgrep header",
            "the ripgrep overlay writes a header",
            "the kubernetes operator reconciles a deployment",
        ]
        .iter()
        .map(|text| doc(text))
        .collect();
        let query = doc("parser");
        let first = index.score(&query);
        // Only the atom carrying the word survives the first pass.
        assert_eq!(first.len(), 1);
        let feedback: Vec<(&[String], f64)> = first
            .iter()
            .map(|(ordinal, score)| (documents[*ordinal].as_slice(), *score))
            .collect();
        let expanded = index.expand(&query, &feedback, 10, 0.5);
        assert!(
            expanded.iter().any(|(term, _)| term == "ripgrep"),
            "{expanded:?}"
        );
        let second = index.score_weighted(&expanded);
        // The overlay shares no word with the question and is reached anyway.
        assert!(
            second.iter().any(|(ordinal, _)| *ordinal == 1),
            "{second:?}"
        );
        // And the unrelated atom still is not.
        assert!(
            !second.iter().any(|(ordinal, _)| *ordinal == 2),
            "{second:?}"
        );
    }

    /// Feedback taken on trust is how this drifts, so the words asked for keep
    /// their share whatever the first pass returned.
    #[test]
    fn the_words_asked_for_keep_their_share() {
        let index = corpus(&["parser header", "overlay record"]);
        let documents: Vec<Vec<String>> = ["parser header", "overlay record"]
            .iter()
            .map(|text| doc(text))
            .collect();
        let query = doc("parser");
        let feedback: Vec<(&[String], f64)> = vec![(documents[1].as_slice(), 1.0)];
        let expanded = index.expand(&query, &feedback, 10, 0.5);
        let asked: f64 = expanded
            .iter()
            .filter(|(term, _)| term == "parser")
            .map(|(_, weight)| *weight)
            .sum();
        assert!((asked - 0.5).abs() < 1e-9, "{expanded:?}");
        let guessed: f64 = expanded
            .iter()
            .filter(|(term, _)| term != "parser")
            .map(|(_, weight)| *weight)
            .sum();
        assert!((guessed - 0.5).abs() < 1e-9, "{expanded:?}");
    }

    /// With nothing to learn from, the expanded query is the query.
    #[test]
    fn no_feedback_leaves_the_query_alone() {
        let index = corpus(&["parser header", "overlay record"]);
        let query = doc("parser header");
        let expanded = index.expand(&query, &[], 10, 0.5);
        assert_eq!(expanded.len(), 2);
        let plain = index.score(&query);
        let weighted = index.score_weighted(&expanded);
        assert_eq!(plain.len(), weighted.len());
        // Same ordering, since every term was scaled by the same share.
        let best = |scored: &[(usize, f64)]| {
            scored
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(ordinal, _)| *ordinal)
        };
        assert_eq!(best(&plain), best(&weighted));
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
