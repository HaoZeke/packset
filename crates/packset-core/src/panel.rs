//! Named fuse then diversify. Host picks the sequence.
//!
//! Default fuse is Borda. Default diversify is MMR. Unknown names
//! fail closed. Later voters add a variant and a match arm; they
//! do not change the default.

use std::hash::Hash;

use crate::borda::{borda_merge, Ballot};
use crate::mmr::{mmr_rerank, Ranked};
use crate::rrf::rrf_merge;
use crate::tideman::ranked_pairs_merge;

/// Fuse slot. Only implemented names parse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Fuse {
    #[default]
    Borda,
    Rrf,
    Tideman,
}

/// Diversify slot. `None` keeps fuse order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Diversify {
    #[default]
    Mmr,
    None,
}

/// Host sequence. Clients do not choose this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Panel {
    pub fuse: Fuse,
    pub diversify: Diversify,
}

/// A name that is not an implemented voter.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnknownVoter {
    #[error("unknown fuse `{0}`")]
    Fuse(String),
    #[error("unknown diversify `{0}`")]
    Diversify(String),
}

impl Fuse {
    pub fn parse(name: &str) -> Result<Self, UnknownVoter> {
        match name {
            "borda" => Ok(Self::Borda),
            "rrf" => Ok(Self::Rrf),
            "tideman" => Ok(Self::Tideman),
            other => Err(UnknownVoter::Fuse(other.to_string())),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Borda => "borda",
            Self::Rrf => "rrf",
            Self::Tideman => "tideman",
        }
    }
}

impl Diversify {
    pub fn parse(name: &str) -> Result<Self, UnknownVoter> {
        match name {
            "mmr" => Ok(Self::Mmr),
            "none" => Ok(Self::None),
            other => Err(UnknownVoter::Diversify(other.to_string())),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mmr => "mmr",
            Self::None => "none",
        }
    }
}

impl Default for Panel {
    fn default() -> Self {
        Self {
            fuse: Fuse::Borda,
            diversify: Diversify::Mmr,
        }
    }
}

impl Panel {
    pub fn parse(fuse: &str, diversify: &str) -> Result<Self, UnknownVoter> {
        Ok(Self {
            fuse: Fuse::parse(fuse)?,
            diversify: Diversify::parse(diversify)?,
        })
    }

    pub fn fuse_merge<T>(&self, ballots: &[Ballot<T>], k: usize) -> Vec<T>
    where
        T: Clone + Eq + Hash,
    {
        match self.fuse {
            Fuse::Borda => borda_merge(ballots, k),
            Fuse::Rrf => {
                let mut out = rrf_merge(ballots, 60);
                out.truncate(k);
                out
            }
            Fuse::Tideman => ranked_pairs_merge(ballots, k),
        }
    }

    pub fn rerank(&self, items: &[Ranked], lambda: f64) -> Vec<String> {
        match self.diversify {
            Diversify::Mmr => mmr_rerank(items, lambda),
            Diversify::None => items.iter().map(|i| i.id.clone()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keep_dup_other() -> Vec<Ranked> {
        vec![
            Ranked {
                id: "keep".into(),
                rel: 1.0,
                tokens: ["review", "open", "repro"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            },
            Ranked {
                id: "dup".into(),
                rel: 0.55,
                tokens: ["review", "open", "repro"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            },
            Ranked {
                id: "other".into(),
                rel: 0.5,
                tokens: ["pin", "zircon", "index"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            },
        ]
    }

    #[test]
    fn default_is_borda_then_mmr() {
        let panel = Panel::default();
        assert_eq!(panel.fuse, Fuse::Borda);
        assert_eq!(panel.diversify, Diversify::Mmr);
        assert_eq!(panel.fuse.as_str(), "borda");
        assert_eq!(panel.diversify.as_str(), "mmr");
        assert_eq!(Panel::parse("borda", "mmr").unwrap(), panel);
    }

    #[test]
    fn default_fuse_matches_borda_fixture() {
        let a = vec!["x", "y", "z"];
        let b = vec!["y", "x", "z"];
        let out = Panel::default().fuse_merge(&[a, b], 3);
        assert_eq!(out, vec!["x", "y", "z"]);
        let a = vec!["a", "b", "c"];
        let b = vec!["b", "c", "a"];
        let out = Panel::default().fuse_merge(&[a, b], 3);
        assert_eq!(out, vec!["b", "a", "c"]);
    }

    #[test]
    fn default_rerank_matches_mmr_fixture() {
        let items = keep_dup_other();
        let reranked = Panel::default().rerank(&items, 0.7);
        assert_eq!(reranked, mmr_rerank(&items, 0.7));
        assert_eq!(reranked, vec!["keep", "other", "dup"]);
    }

    #[test]
    fn diversify_none_keeps_fuse_order() {
        let panel = Panel {
            fuse: Fuse::Borda,
            diversify: Diversify::None,
        };
        let items = keep_dup_other();
        assert_eq!(panel.rerank(&items, 0.7), vec!["keep", "dup", "other"]);
    }

    #[test]
    fn parse_rrf_calls_rrf_merge() {
        assert_eq!(Fuse::parse("rrf").unwrap(), Fuse::Rrf);
        assert_eq!(Fuse::Rrf.as_str(), "rrf");
        let panel = Panel::parse("rrf", "mmr").unwrap();
        assert_eq!(panel.fuse, Fuse::Rrf);
        assert_eq!(panel.diversify, Diversify::Mmr);
        let a = vec!["x", "y", "z"];
        let b = vec!["y", "x", "z"];
        let c = vec!["z"];
        let out = Panel {
            fuse: Fuse::Rrf,
            diversify: Diversify::None,
        }
        .fuse_merge(&[a, b, c], 3);
        assert_eq!(out[0], "z");
    }

    #[test]
    fn parse_tideman_calls_ranked_pairs_merge() {
        assert_eq!(Fuse::parse("tideman").unwrap(), Fuse::Tideman);
        assert_eq!(Fuse::Tideman.as_str(), "tideman");
        let panel = Panel::parse("tideman", "mmr").unwrap();
        assert_eq!(panel.fuse, Fuse::Tideman);
        assert_eq!(panel.diversify, Diversify::Mmr);
        let a = vec!["a", "c", "d", "e", "f"];
        let b = vec!["a", "c", "d", "e", "f"];
        let c = vec!["a", "c", "d", "e", "f"];
        let d = vec!["c", "d", "e", "f", "a"];
        let e = vec!["d", "e", "f", "c", "a"];
        let out = Panel {
            fuse: Fuse::Tideman,
            diversify: Diversify::None,
        }
        .fuse_merge(&[a, b, c, d, e], 5);
        assert_eq!(out[0], "a");
    }

    #[test]
    fn unknown_name_is_error() {
        assert!(matches!(
            Fuse::parse("not-a-voter"),
            Err(UnknownVoter::Fuse(name)) if name == "not-a-voter"
        ));
        assert!(matches!(
            Diversify::parse("not-a-voter"),
            Err(UnknownVoter::Diversify(name)) if name == "not-a-voter"
        ));
        assert!(Panel::parse("borda", "not-a-voter").is_err());
        assert!(Fuse::parse("").is_err());
        assert!(Fuse::parse("Borda").is_err());
        assert!(Fuse::parse("Tideman").is_err());
        assert!(Fuse::parse("ranked-pairs").is_err());
    }
}
