//! The loopback pack writer.
//!
//! One process owns `memory.lmdb` and every client speaks HTTP to it, so an
//! isolated harness home does not get a private store. Cards stay files
//! because a person edits them; atoms are a database because a program does.

pub mod cards;
pub mod home;
pub mod service;
pub mod store;

pub use home::Home;
pub use service::Service;
pub use store::Store;
