//! Store-agnostic packset algorithms.
//!
//! Host merge is Borda (`k - position`). MMR and temporal decay
//! are voters / post-merge, not a second store.

pub mod atom;
pub mod borda;
pub mod decay;
pub mod extract;
pub mod mmr;

pub use atom::{entity_jaccard, is_live, SCHEMA as ATOM_SCHEMA};
pub use borda::{borda_merge, Ballot};
pub use decay::temporal_decay;
pub use extract::{claim_from_user, is_tool_dump};
pub use mmr::mmr_rerank;
