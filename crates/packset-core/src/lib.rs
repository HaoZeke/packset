//! Store-agnostic packset algorithms.
//!
//! Host merge is a named fuse then diversify panel. Default fuse
//! is Borda (`k - position`). Reciprocal Rank Fusion and Schulze
//! beatpath are named fuse voters. Default diversify is MMR.
//! Temporal decay is a voter, not a second store.

pub mod atom;
pub mod borda;
pub mod decay;
pub mod extract;
pub mod mmr;
pub mod panel;
pub mod rrf;
pub mod schulze;

pub use atom::{entity_jaccard, is_live, SCHEMA as ATOM_SCHEMA};
pub use borda::{borda_merge, Ballot};
pub use decay::temporal_decay;
pub use extract::{claim_from_user, is_tool_dump};
pub use mmr::mmr_rerank;
pub use panel::{Diversify, Fuse, Panel, UnknownVoter};
pub use rrf::rrf_merge;
pub use schulze::schulze_merge;
