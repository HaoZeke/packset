//! Named fuse then diversify then decay. Host picks the sequence.
//!
//! Default fuse is Borda. Default diversify is MMR. Default decay
//! is off. Names come from `PACKSET_FUSE`, `PACKSET_DIVERSIFY`,
//! and `PACKSET_DECAY`, or from packsetd flags. Clients do not
//! choose this. Unknown and reserved-unimplemented names fail
//! closed. Later voters add a variant and a match arm; they do
//! not change the default.

use std::env;
use std::hash::Hash;

use crate::borda::{borda_merge, Ballot};
use crate::decay::temporal_decay;
use crate::mmr::{mmr_rerank, Ranked};
use crate::rrf::rrf_merge;

/// Half-life used when decay is on. Matches the host recency scale.
const DECAY_HALF_LIFE_DAYS: f64 = 14.0;

/// Fuse slot. Only implemented names parse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Fuse {
    #[default]
    Borda,
    Rrf,
}

/// Diversify slot. `None` keeps fuse order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Diversify {
    #[default]
    Mmr,
    None,
}

/// Decay slot. Off leaves fuse scores unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Decay {
    #[default]
    Off,
    On,
}

/// Host sequence. Clients do not choose this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Panel {
    pub fuse: Fuse,
    pub diversify: Diversify,
    pub decay: Decay,
}

/// A name that is not an implemented voter.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnknownVoter {
    #[error("unknown fuse `{0}`")]
    Fuse(String),
    #[error("unknown diversify `{0}`")]
    Diversify(String),
    #[error("unknown decay `{0}`")]
    Decay(String),
    #[error("not implemented fuse `{0}`")]
    FuseNotImplemented(String),
    #[error("not implemented diversify `{0}`")]
    DiversifyNotImplemented(String),
}

fn env_or<'a>(
    raw: Option<&'a str>,
    default: &'a str,
    empty: fn(String) -> UnknownVoter,
) -> Result<&'a str, UnknownVoter> {
    match raw {
        None => Ok(default),
        Some("") => Err(empty(String::new())),
        Some(name) => Ok(name),
    }
}

impl Fuse {
    pub fn parse(name: &str) -> Result<Self, UnknownVoter> {
        match name {
            "borda" => Ok(Self::Borda),
            "rrf" => Ok(Self::Rrf),
            "dowdall" | "combmnz" | "kemeny" | "schulze" | "copeland" | "tideman" => {
                Err(UnknownVoter::FuseNotImplemented(name.to_string()))
            }
            other => Err(UnknownVoter::Fuse(other.to_string())),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Borda => "borda",
            Self::Rrf => "rrf",
        }
    }
}

impl Diversify {
    pub fn parse(name: &str) -> Result<Self, UnknownVoter> {
        match name {
            "mmr" => Ok(Self::Mmr),
            "none" => Ok(Self::None),
            "dpp" => Err(UnknownVoter::DiversifyNotImplemented(name.to_string())),
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

impl Decay {
    pub fn parse(name: &str) -> Result<Self, UnknownVoter> {
        match name {
            "off" => Ok(Self::Off),
            "on" => Ok(Self::On),
            other => Err(UnknownVoter::Decay(other.to_string())),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
        }
    }
}

impl Default for Panel {
    fn default() -> Self {
        Self {
            fuse: Fuse::Borda,
            diversify: Diversify::Mmr,
            decay: Decay::Off,
        }
    }
}

impl Panel {
    pub fn parse(fuse: &str, diversify: &str) -> Result<Self, UnknownVoter> {
        Self::named(fuse, diversify, "off")
    }

    pub fn named(fuse: &str, diversify: &str, decay: &str) -> Result<Self, UnknownVoter> {
        Ok(Self {
            fuse: Fuse::parse(fuse)?,
            diversify: Diversify::parse(diversify)?,
            decay: Decay::parse(decay)?,
        })
    }

    /// Read `PACKSET_FUSE`, `PACKSET_DIVERSIFY`, `PACKSET_DECAY`.
    /// Unset keys take the default. Empty values fail closed.
    pub fn from_env() -> Result<Self, UnknownVoter> {
        let fuse = env::var("PACKSET_FUSE").ok();
        let diversify = env::var("PACKSET_DIVERSIFY").ok();
        let decay = env::var("PACKSET_DECAY").ok();
        Self::from_env_vars(fuse.as_deref(), diversify.as_deref(), decay.as_deref())
    }

    pub fn from_env_vars(
        fuse: Option<&str>,
        diversify: Option<&str>,
        decay: Option<&str>,
    ) -> Result<Self, UnknownVoter> {
        Self::named(
            env_or(fuse, "borda", UnknownVoter::Fuse)?,
            env_or(diversify, "mmr", UnknownVoter::Diversify)?,
            env_or(decay, "off", UnknownVoter::Decay)?,
        )
    }

    pub fn decay_weight(&self, source: &str, age_days: f64) -> f64 {
        match self.decay {
            Decay::Off => 1.0,
            Decay::On => temporal_decay(source, age_days, Some(DECAY_HALF_LIFE_DAYS)),
        }
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
        assert_eq!(panel.decay, Decay::Off);
        assert_eq!(panel.fuse.as_str(), "borda");
        assert_eq!(panel.diversify.as_str(), "mmr");
        assert_eq!(panel.decay.as_str(), "off");
        assert_eq!(Panel::parse("borda", "mmr").unwrap(), panel);
        assert_eq!(Panel::named("borda", "mmr", "off").unwrap(), panel);
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
            decay: Decay::Off,
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
            decay: Decay::Off,
        }
        .fuse_merge(&[a, b, c], 3);
        assert_eq!(out[0], "z");
    }

    #[test]
    fn from_env_vars_reads_named_sequence() {
        let panel = Panel::from_env_vars(Some("rrf"), Some("none"), Some("on")).unwrap();
        assert_eq!(panel.fuse, Fuse::Rrf);
        assert_eq!(panel.diversify, Diversify::None);
        assert_eq!(panel.decay, Decay::On);
        let unset = Panel::from_env_vars(None, None, None).unwrap();
        assert_eq!(unset, Panel::default());
        assert!(Panel::from_env_vars(Some(""), None, None).is_err());
        let rrf = Panel::from_env_vars(Some("rrf"), None, None).unwrap();
        let a = vec!["x", "y", "z"];
        let b = vec!["y", "x", "z"];
        let c = vec!["z"];
        let out = rrf.fuse_merge(&[a, b, c], 3);
        assert_eq!(out[0], "z");
    }

    #[test]
    fn decay_off_is_one_on_uses_temporal() {
        let off = Panel::default();
        assert_eq!(off.decay_weight("session", 14.0), 1.0);
        let on = Panel::named("borda", "mmr", "on").unwrap();
        let w = on.decay_weight("session", 14.0);
        assert!((w - 0.5).abs() < 1e-9);
        assert_eq!(on.decay_weight("global", 400.0), 1.0);
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
        assert!(matches!(
            Decay::parse("maybe"),
            Err(UnknownVoter::Decay(name)) if name == "maybe"
        ));
        assert!(Panel::parse("borda", "not-a-voter").is_err());
        assert!(Panel::named("borda", "mmr", "maybe").is_err());
        assert!(Fuse::parse("").is_err());
        assert!(Fuse::parse("Borda").is_err());
        assert!(Decay::parse("On").is_err());
    }

    #[test]
    fn reserved_name_is_not_implemented() {
        for name in [
            "dowdall", "combmnz", "kemeny", "schulze", "copeland", "tideman",
        ] {
            assert!(
                matches!(
                    Fuse::parse(name),
                    Err(UnknownVoter::FuseNotImplemented(got)) if got == name
                ),
                "{name}"
            );
        }
        assert!(matches!(
            Diversify::parse("dpp"),
            Err(UnknownVoter::DiversifyNotImplemented(name)) if name == "dpp"
        ));
        assert!(Panel::from_env_vars(Some("kemeny"), None, None).is_err());
        assert!(Panel::from_env_vars(None, Some("dpp"), None).is_err());
    }
}
