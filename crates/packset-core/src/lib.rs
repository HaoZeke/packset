//! Store-agnostic packset algorithms.
//!
//! Host merge is a named fuse then diversify then decay panel.
//! Default fuse is Borda (`k - position`). Reciprocal Rank Fusion
//! is a named fuse voter. Default diversify is MMR. Default decay
//! is off. The host reads `PACKSET_FUSE`, `PACKSET_DIVERSIFY`,
//! and `PACKSET_DECAY`. Clients do not choose this.

pub mod atom;
pub mod borda;
pub mod decay;
pub mod extract;
pub mod mmr;
pub mod panel;
pub mod rrf;

pub use atom::{entity_jaccard, is_live, SCHEMA as ATOM_SCHEMA};
pub use borda::{borda_merge, Ballot};
pub use decay::temporal_decay;
pub use extract::{claim_from_user, is_tool_dump};
pub use mmr::mmr_rerank;
pub use panel::{Decay, Diversify, Fuse, Panel, UnknownVoter};
pub use rrf::rrf_merge;
