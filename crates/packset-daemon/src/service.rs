//! What the writer does, apart from how a client asked.
//!
//! The HTTP layer decodes and encodes; everything a verb actually means lives
//! here, so the rules can be tested without a socket.

use std::collections::BTreeMap;
use std::sync::Mutex;

use packset_core::clock;
use packset_core::record::{self, AtomError, MEMORY_CAP, USER_CAP};
use serde_json::{json, Map, Value};

use crate::cards;
use crate::home::Home;
use crate::store::{Record, Store};

/// The cap on one attached body.
pub const ATTACH_CAP: usize = 200_000;

/// One workspace's pending attachment.
#[derive(Debug, Clone, Default)]
pub struct Attachment {
    /// The body.
    pub text: String,
    /// What it is, for a reader.
    pub label: String,
}

/// The writer: the store, the cards, and the one-shot attach slots.
pub struct Service {
    home: Home,
    store: Store,
    attach: Mutex<BTreeMap<String, Attachment>>,
}

impl Service {
    /// Open the pack at `home`.
    ///
    /// # Errors
    ///
    /// Fails when the store cannot be opened or the lock is held.
    pub fn open(home: Home) -> anyhow::Result<Self> {
        let store = Store::open(home.root())?;
        Ok(Self {
            home,
            store,
            attach: Mutex::new(BTreeMap::new()),
        })
    }

    /// The pack home.
    #[must_use]
    pub fn home(&self) -> &Home {
        &self.home
    }

    /// The atom store.
    #[must_use]
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Store one atom, or return the live one that already says it.
    ///
    /// Deduplication is on text, kind and set together, and it is what makes
    /// `Remember:` safe to send twice: a client that retries does not get two
    /// atoms saying one thing.
    ///
    /// # Errors
    ///
    /// [`AtomError`] when the text is a tool dump or the record does not
    /// validate, else the store's.
    pub fn add(&self, mut atom: Record) -> anyhow::Result<Record> {
        let text = atom
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if packset_core::extract::is_tool_dump(&text) {
            anyhow::bail!(AtomError("tool dump is attach, not an atom".into()));
        }
        record::validate(&mut atom).map_err(anyhow::Error::new)?;

        if atom
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .is_empty()
        {
            atom.insert("id".into(), Value::String(new_id()));
        }
        if atom
            .get("ts")
            .and_then(Value::as_str)
            .unwrap_or("")
            .is_empty()
        {
            atom.insert("ts".into(), Value::String(clock::utcnow()));
        }
        atom.insert("tombstone".into(), Value::Bool(false));
        atom.entry("embedding").or_insert(Value::Null);

        let workspace = atom
            .get("workspace")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let named = atom.get("set").and_then(Value::as_str).map(str::to_string);

        // A set-scoped atom is compared against its own set; an unscoped one
        // against the unscoped atoms, so pinning a set does not make a claim
        // look like a duplicate of one in another scope.
        let live: Vec<Record> = match named.as_deref() {
            Some(name) => self.store.current(&workspace, Some(name))?,
            None => self
                .store
                .current(&workspace, None)?
                .into_iter()
                .filter(|peer| peer.get("set").is_none())
                .collect(),
        };
        for existing in &live {
            if existing.get("text") == atom.get("text")
                && existing.get("kind") == atom.get("kind")
                && existing.get("set") == atom.get("set")
            {
                return Ok(existing.clone());
            }
        }

        let now = clock::utcnow();
        let mut batch = Vec::new();
        if record::is_live(&atom, &now) {
            let rewritten = record::apply_links(&mut atom, &live, record::LINK_THRESHOLD, &now);
            for mut peer in rewritten {
                peer.insert("ts".into(), Value::String(clock::utcnow()));
                batch.push(peer);
            }
        } else if !atom.contains_key("links") {
            atom.insert("links".into(), Value::Array(Vec::new()));
        }
        let mut all = vec![atom.clone()];
        all.append(&mut batch);
        self.store.upsert_many(&all)?;
        self.project_atoms(&all);
        Ok(atom)
    }

    /// Merge `fields` into one current atom.
    ///
    /// # Errors
    ///
    /// [`AtomError`] when the id is not current or the result does not
    /// validate, else the store's.
    pub fn update(
        &self,
        workspace: &str,
        id: &str,
        fields: &Map<String, Value>,
    ) -> anyhow::Result<Record> {
        let current = self.store.current(workspace, None)?;
        let mut updated = current
            .iter()
            .find(|a| a.get("id").and_then(Value::as_str) == Some(id))
            .cloned()
            .ok_or_else(|| anyhow::Error::new(AtomError(format!("no current atom {id}"))))?;
        for (key, value) in fields {
            updated.insert(key.clone(), value.clone());
        }
        updated.insert("id".into(), Value::String(id.to_string()));
        updated.insert("workspace".into(), Value::String(workspace.to_string()));
        updated.insert("ts".into(), Value::String(clock::utcnow()));
        updated.insert("tombstone".into(), Value::Bool(false));
        record::validate(&mut updated).map_err(anyhow::Error::new)?;

        let now = clock::utcnow();
        let mut batch = Vec::new();
        if record::is_live(&updated, &now) {
            let rewritten =
                record::apply_links(&mut updated, &current, record::LINK_THRESHOLD, &now);
            for mut peer in rewritten {
                peer.insert("ts".into(), Value::String(clock::utcnow()));
                batch.push(peer);
            }
        } else if !updated.contains_key("links") {
            updated.insert("links".into(), Value::Array(Vec::new()));
        }
        let mut all = vec![updated.clone()];
        all.append(&mut batch);
        self.store.upsert_many(&all)?;
        self.project_atoms(&all);
        Ok(updated)
    }

    /// Move one atom along the review clock.
    ///
    /// # Errors
    ///
    /// As [`Service::update`].
    pub fn grade(&self, workspace: &str, id: &str, recalled: bool) -> anyhow::Result<Record> {
        let mut atom = self
            .store
            .current(workspace, None)?
            .into_iter()
            .find(|a| a.get("id").and_then(Value::as_str) == Some(id))
            .ok_or_else(|| anyhow::Error::new(AtomError(format!("no current atom {id}"))))?;
        let grade = if recalled {
            record::Grade::Recalled
        } else {
            record::Grade::Lapsed
        };
        record::schedule_review(&mut atom, &clock::utcnow(), grade, None);
        let mut fields = Map::new();
        fields.insert("due_at".into(), atom["due_at"].clone());
        fields.insert("review".into(), atom["review"].clone());
        self.update(workspace, id, &fields)
    }

    /// Tombstone one atom and drop it from the projection.
    ///
    /// # Errors
    ///
    /// The store's.
    pub fn delete_atom(&self, workspace: &str, id: &str) -> anyhow::Result<Record> {
        let tomb = self.store.delete(workspace, id)?;
        let _ = crate::milli::delete(&[id.to_string()], &self.home.milli_dir());
        Ok(tomb)
    }

    /// The workspace pack, or the same shape scoped to one set.
    ///
    /// Always `user` / `memory` / `atoms`, so a client splices one shape
    /// whether or not a set is pinned.
    ///
    /// # Errors
    ///
    /// [`AtomError`] for a bad set name, else the store's.
    pub fn pack(&self, workspace: &str, set: Option<&str>) -> anyhow::Result<Value> {
        match set {
            Some(raw) => {
                let named = packset_core::set_name::check(raw)
                    .map_err(|e| anyhow::Error::new(AtomError(e)))?;
                Ok(json!({
                    "workspace": workspace,
                    "set": named,
                    "user": cards::read_text(&self.home.set_user_path(workspace, &named)),
                    "memory": cards::read_text(&self.home.set_memory_path(workspace, &named)),
                    "instructions": cards::read_text(
                        &self.home.set_instructions_path(workspace, &named)
                    ),
                    "atoms": self.store.current(workspace, Some(&named))?,
                }))
            }
            None => Ok(json!({
                "workspace": workspace,
                "user": cards::read_text(&self.home.user_path()),
                "memory": cards::read_text(&self.home.memory_path(workspace)),
                "atoms": self.store.current(workspace, None)?,
            })),
        }
    }

    /// The active set for a workspace, or empty when nothing is pinned.
    #[must_use]
    pub fn pin(&self, workspace: &str) -> String {
        let raw = cards::read_text(&self.home.pin_path(workspace));
        raw.lines()
            .next()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .and_then(|line| packset_core::set_name::check(line).ok())
            .unwrap_or_default()
    }

    /// Pin a set, or clear the pin when the name is empty.
    ///
    /// # Errors
    ///
    /// [`AtomError`] for a bad name, else the write's.
    pub fn set_pin(&self, workspace: &str, name: &str) -> anyhow::Result<String> {
        let path = self.home.pin_path(workspace);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if name.trim().is_empty() {
            std::fs::write(&path, "")?;
            return Ok(String::new());
        }
        let stored =
            packset_core::set_name::check(name).map_err(|e| anyhow::Error::new(AtomError(e)))?;
        std::fs::write(&path, format!("{stored}\n"))?;
        Ok(stored)
    }

    /// The pin plus that set's standing instructions.
    ///
    /// # Errors
    ///
    /// Never, in practice: an unreadable set reads as empty.
    pub fn pin_payload(&self, workspace: &str) -> anyhow::Result<Value> {
        let named = self.pin(workspace);
        let instructions = if named.is_empty() {
            String::new()
        } else {
            cards::read_text(&self.home.set_instructions_path(workspace, &named))
        };
        Ok(json!({
            "workspace": workspace,
            "set": named,
            "instructions": instructions,
        }))
    }

    /// Write whichever of a set's three cards the body carried.
    ///
    /// Absent and empty are different: a key that is not there leaves that card
    /// alone, and a key set to an empty string clears it.
    ///
    /// # Errors
    ///
    /// [`AtomError`] for a bad set name, else the write's.
    pub fn write_set(
        &self,
        workspace: &str,
        name: &str,
        body: &Map<String, Value>,
    ) -> Result<String, cards::WriteError> {
        let stored = packset_core::set_name::check(name).map_err(AtomError)?;
        for (key, path) in [
            ("user", self.home.set_user_path(workspace, &stored)),
            ("memory", self.home.set_memory_path(workspace, &stored)),
            (
                "instructions",
                self.home.set_instructions_path(workspace, &stored),
            ),
        ] {
            let Some(value) = body.get(key) else { continue };
            let text = value.as_str().unwrap_or("");
            let cap = if key == "memory" {
                MEMORY_CAP
            } else {
                USER_CAP
            };
            cards::write_capped(&path, text, cap)?;
        }
        Ok(stored)
    }

    /// Write the seat card, archiving and refusing on overflow.
    ///
    /// # Errors
    ///
    /// The write's, with overflow distinguishable so the caller can answer 413.
    pub fn set_user(&self, text: &str) -> Result<(), cards::WriteError> {
        match cards::write_capped(&self.home.user_path(), text, USER_CAP) {
            Err(cards::WriteError::Overflow(o)) => {
                // Archived first, so the text is refused rather than lost and
                // the day file is what the miner reads. An archive that itself
                // fails is reported instead of the overflow, which is what the
                // writer being replaced does.
                self.archive("global", text)?;
                Err(cards::WriteError::Overflow(o))
            }
            Ok(()) => {
                self.project_cards(None);
                Ok(())
            }
            other => other,
        }
    }

    /// Write a workspace card, archiving and refusing on overflow.
    ///
    /// # Errors
    ///
    /// As [`Service::set_user`].
    pub fn set_memory(&self, workspace: &str, text: &str) -> Result<(), cards::WriteError> {
        match cards::write_capped(&self.home.memory_path(workspace), text, MEMORY_CAP) {
            Err(cards::WriteError::Overflow(o)) => {
                self.archive(workspace, text)?;
                Err(cards::WriteError::Overflow(o))
            }
            Ok(()) => {
                self.project_cards(Some(workspace));
                Ok(())
            }
            other => other,
        }
    }

    /// Append to today's archive for a workspace.
    ///
    /// # Errors
    ///
    /// The write's.
    pub fn archive(&self, workspace: &str, text: &str) -> Result<(), cards::WriteError> {
        let day = clock::utcnow()[..10].to_string();
        let path = self.home.archive_path(workspace, &day);
        cards::add_entry(&path, text, 1_000_000)
    }

    /// Hold one body for a workspace, capped.
    pub fn put_attach(&self, workspace: &str, text: &str, label: &str) -> Value {
        let mut body = text.to_string();
        if body.chars().count() > ATTACH_CAP {
            body = body.chars().take(ATTACH_CAP).collect();
        }
        let slot = Attachment {
            text: body,
            label: label.trim().to_string(),
        };
        let mut held = self.attach.lock().expect("attach lock");
        held.insert(workspace.to_string(), slot.clone());
        json!({"workspace": workspace, "text": slot.text, "label": slot.label})
    }

    /// Take the held body, leaving the slot empty.
    ///
    /// It is one-shot because an attachment is context for the next turn, and a
    /// body that stayed would be spliced into every turn after it.
    pub fn take_attach(&self, workspace: &str) -> Option<Attachment> {
        let mut held = self.attach.lock().expect("attach lock");
        held.remove(workspace)
    }

    /// Read the held body without taking it.
    pub fn peek_attach(&self, workspace: &str) -> Option<Attachment> {
        let held = self.attach.lock().expect("attach lock");
        held.get(workspace).cloned()
    }

    /// Keep the projection level with a write.
    ///
    /// A live atom is upserted and one that has left the live set is deleted,
    /// so a search never ranks something a reader can no longer be shown. With
    /// no search binary on the seat this is a no-op.
    fn project_atoms(&self, atoms: &[Record]) {
        let dir = self.home.milli_dir();
        let now = clock::utcnow();
        let mut live = Vec::new();
        let mut dead = Vec::new();
        for atom in atoms {
            let Some(id) = atom.get("id").and_then(Value::as_str) else {
                continue;
            };
            if record::is_live(atom, &now) || record::is_due(atom, &now) {
                live.push(crate::milli::atom_document(atom));
            } else {
                dead.push(id.to_string());
            }
        }
        if !live.is_empty() {
            let _ = crate::milli::upsert(&live, &dir);
        }
        if !dead.is_empty() {
            let _ = crate::milli::delete(&dead, &dir);
        }
    }

    /// Keep the projection's copy of the cards level with a write.
    fn project_cards(&self, workspace: Option<&str>) {
        let dir = self.home.milli_dir();
        let workspace = workspace.unwrap_or("");
        let user = cards::read_text(&self.home.user_path());
        let memory = if workspace.is_empty() {
            String::new()
        } else {
            cards::read_text(&self.home.memory_path(workspace))
        };
        let docs = crate::milli::pack_documents(workspace, &user, &memory, &[]);
        if !docs.is_empty() {
            let _ = crate::milli::upsert(&docs, &dir);
        }
    }

    /// Ranked hits, and which engine produced them.
    ///
    /// The projection answers when it is there and the linear scan otherwise,
    /// and every failure in the projection falls back rather than returning a
    /// partial answer: a wrong answer that looks complete is worse than a
    /// slower one that is right.
    ///
    /// # Errors
    ///
    /// [`AtomError`] for a bad set name, else the store's.
    pub fn search(
        &self,
        workspace: &str,
        query: &str,
        limit: usize,
        set: Option<&str>,
        panel: &packset_core::Panel,
    ) -> anyhow::Result<Value> {
        let named = match set {
            Some(raw) => Some(
                packset_core::set_name::check(raw).map_err(|e| anyhow::Error::new(AtomError(e)))?,
            ),
            None => None,
        };
        let scope = named.as_deref();
        // A named set swaps the prose for that set's cards. The atom list stays
        // the whole live set, because the scope is a filter in the scorer and
        // not a smaller corpus.
        let (user, memory) = match scope {
            Some(name) => (
                cards::read_text(&self.home.set_user_path(workspace, name)),
                cards::read_text(&self.home.set_memory_path(workspace, name)),
            ),
            None => (
                cards::read_text(&self.home.user_path()),
                cards::read_text(&self.home.memory_path(workspace)),
            ),
        };
        let atoms = self.store.current(workspace, None)?;
        let now = clock::utcnow();

        if packset_core::search::tokens(query).is_empty() {
            return Ok(json!({"hits": [], "engine": "linear"}));
        }

        let dir = self.home.milli_dir();
        let corpus = crate::milli::Corpus {
            workspace,
            user: &user,
            memory: &memory,
            atoms: &atoms,
        };
        let projected = crate::milli::search(corpus, query, limit, &dir, scope);
        let (mut ranked, engine) = match projected {
            Some(atom_hits) => {
                // Prose always comes from the pack, so the index copy of a card
                // can be stale without anyone reading it.
                let prose = packset_core::search::search_linear(
                    &user,
                    &memory,
                    &[],
                    query,
                    limit,
                    scope,
                    &now,
                );
                (
                    packset_core::search::merge_ballots(&[prose, atom_hits], limit, panel, &now),
                    "milli",
                )
            }
            None => (
                packset_core::search::search_linear(
                    &user, &memory, &atoms, query, limit, scope, &now,
                ),
                "linear",
            ),
        };
        let due = packset_core::search::due_hits(&atoms, scope, &now);
        if !due.is_empty() {
            ranked = packset_core::search::front_due(due, ranked, limit);
        }
        Ok(json!({"hits": ranked, "engine": engine}))
    }

    /// Mine one archived day into proposals.
    ///
    /// # Errors
    ///
    /// The miner's, or the store's.
    pub fn compact(
        &self,
        workspace: &str,
        day: Option<&str>,
        transcript: Option<&str>,
    ) -> anyhow::Result<Value> {
        let live = self.store.current(workspace, None)?;
        let proposed =
            crate::proposals::compact_day(&self.home, workspace, day, &live, transcript, new_id)?;
        Ok(json!({"n": proposed.len(), "proposals": proposed}))
    }

    /// Propose one claim from one piece of text.
    ///
    /// # Errors
    ///
    /// The miner's, or [`AtomError`] when there is nothing to propose.
    pub fn propose(&self, body: &Map<String, Value>) -> anyhow::Result<Value> {
        let workspace = body
            .get("workspace")
            .and_then(Value::as_str)
            .filter(|w| !w.is_empty())
            .ok_or_else(|| anyhow::Error::new(AtomError("workspace required".into())))?;
        let text = body.get("text").and_then(Value::as_str).unwrap_or("");
        let when = body
            .get("when")
            .and_then(Value::as_str)
            .filter(|w| !w.is_empty())
            .unwrap_or("onDemand");
        let job = body
            .get("job")
            .and_then(Value::as_str)
            .filter(|j| !j.is_empty())
            .unwrap_or("extract");
        let transcript = body.get("transcript").and_then(Value::as_str);
        let live = self.store.current(workspace, None)?;
        let wall = crate::proposals::fence(&self.home, workspace, &live);
        let rec = crate::proposals::propose(
            &self.home,
            crate::proposals::Mining {
                workspace,
                job,
                when,
                wall: &wall,
                transcript,
            },
            text,
            new_id,
        )?;
        rec.ok_or_else(|| anyhow::Error::new(AtomError("nothing to propose".into())))
    }

    /// Turn an accepted proposal into a stored atom.
    ///
    /// # Errors
    ///
    /// The miner's, or the store's.
    pub fn accept(&self, workspace: &str, proposal_id: &str) -> anyhow::Result<Record> {
        let (atom, rec) = crate::proposals::accept(&self.home, workspace, proposal_id)?;
        let stored = self.add(atom)?;
        let atom_id = stored.get("id").and_then(Value::as_str).unwrap_or("");
        crate::proposals::mark_accepted(&self.home, workspace, &rec, atom_id)?;
        Ok(stored)
    }

    /// The open proposals for a workspace.
    #[must_use]
    pub fn proposals(&self, workspace: &str) -> Vec<Value> {
        crate::proposals::list_open(&self.home, workspace)
    }

    /// Seat home, atom counts by kind, pin, index and embedder.
    ///
    /// # Errors
    ///
    /// The scan's.
    pub fn status(&self, workspace: Option<&str>) -> anyhow::Result<Value> {
        let now = clock::utcnow();
        let mut live: BTreeMap<String, usize> = BTreeMap::new();
        let mut tomb: BTreeMap<String, usize> = BTreeMap::new();
        let mut expired: BTreeMap<String, usize> = BTreeMap::new();
        let mut last_write = String::new();
        for rec in self.store.scan(workspace)? {
            let kind = rec
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            if let Some(ts) = rec.get("ts").and_then(Value::as_str) {
                if ts > last_write.as_str() {
                    last_write = ts.to_string();
                }
            }
            if rec
                .get("tombstone")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                *tomb.entry(kind).or_insert(0) += 1;
            } else if record::is_live(&rec, &now) {
                *live.entry(kind).or_insert(0) += 1;
            } else {
                *expired.entry(kind).or_insert(0) += 1;
            }
        }
        let milli_dir = self.home.milli_dir();
        let index_ready = crate::milli::index_ready(&milli_dir);
        let pin = workspace.map(|w| self.pin(w)).unwrap_or_default();
        Ok(json!({
            "home": self.home.root().display().to_string(),
            "workspace": workspace.unwrap_or(""),
            "set": pin,
            "live": live.values().sum::<usize>(),
            "tombstone": tomb.values().sum::<usize>(),
            "expired": expired.values().sum::<usize>(),
            "live_by_kind": live,
            "tombstone_by_kind": tomb,
            "expired_by_kind": expired,
            "last_write_ts": if last_write.is_empty() { Value::Null } else { Value::String(last_write) },
            "milli": {
                "binary": milli_binary(),
                "index_dir": milli_dir.display().to_string(),
                "index_ready": index_ready,
            },
            "embedder": {
                "enabled": embed_enabled(),
                "available": false,
            },
        }))
    }
}

/// Whether the dense-rank embedder is switched on.
///
/// The Rust writer reports it unavailable because the ONNX runtime behind it is
/// not in this process. Keyword search does not depend on it.
fn embed_enabled() -> bool {
    let raw = std::env::var("INSIDE_EMBED").unwrap_or_else(|_| "on".into());
    !matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "0" | "off" | "none" | "false" | "no"
    )
}

/// The search binary as `/v1/status` reports it.
///
/// One lookup, shared with the search path, so status cannot say the
/// projection is available while search fails to find it.
fn milli_binary() -> Value {
    crate::milli::binary().map_or(Value::Null, |p| Value::String(p.display().to_string()))
}

/// A fresh atom id: thirty-two hex characters, the shape already in the store.
fn new_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0u128, |d| d.as_nanos());
    let pid = u128::from(std::process::id());
    let counter = {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        u128::from(NEXT.fetch_add(1, Ordering::Relaxed))
    };
    // Not a v4 uuid and not claiming to be: unique on this seat is the whole
    // requirement, and the store keys on workspace and id together.
    let mut state = nanos ^ (pid << 64) ^ (counter << 32);
    let mut out = String::with_capacity(32);
    for _ in 0..32 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let nibble = ((state >> 64) & 0xf) as u8;
        out.push(char::from_digit(u32::from(nibble), 16).unwrap_or('0'));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn service() -> (tempfile::TempDir, Service) {
        let dir = tempfile::tempdir().unwrap();
        let svc = Service::open(Home::new(dir.path())).unwrap();
        (dir, svc)
    }

    fn atom(text: &str) -> Record {
        json!({
            "workspace": "w",
            "text": text,
            "kind": "voice",
            "about_peer": "rgoswami",
            "by_peer": "hermes"
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[test]
    fn an_id_is_thirty_two_hex_characters_and_does_not_repeat() {
        let a = new_id();
        assert_eq!(a.len(), 32, "{a}");
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()), "{a}");
        let many: std::collections::HashSet<String> = (0..1000).map(|_| new_id()).collect();
        assert_eq!(many.len(), 1000, "ids collided");
    }

    #[test]
    fn the_same_claim_twice_is_one_atom() {
        let (_dir, svc) = service();
        let first = svc.add(atom("Reviews open with a check.")).unwrap();
        let again = svc.add(atom("Reviews open with a check.")).unwrap();
        assert_eq!(first["id"], again["id"], "a retry is not a second claim");
        assert_eq!(svc.store().current("w", None).unwrap().len(), 1);
    }

    #[test]
    fn a_set_scoped_claim_is_not_a_duplicate_of_an_unscoped_one() {
        let (_dir, svc) = service();
        let plain = svc.add(atom("Reviews open with a check.")).unwrap();
        let mut scoped = atom("Reviews open with a check.");
        scoped.insert("set".into(), json!("review"));
        let scoped = svc.add(scoped).unwrap();
        assert_ne!(plain["id"], scoped["id"]);
        assert_eq!(svc.store().current("w", None).unwrap().len(), 2);
    }

    #[test]
    fn a_tool_dump_is_refused_as_an_atom() {
        let (_dir, svc) = service();
        let listing = std::iter::once("total 48".to_string())
            .chain((0..7).map(|i| format!("-rw-r--r-- 1 x x 0 Jan 1 00:00 file{i}")))
            .collect::<Vec<_>>()
            .join("\n");
        let err = svc.add(atom(&listing)).unwrap_err();
        assert!(err.to_string().contains("attach"), "{err}");

        // A fenced capture naming a stream is the other shape.
        let fenced = atom("Here is the run:\n```\nstdout: everything fine\n```");
        assert!(svc.add(fenced).is_err());
    }

    #[test]
    fn linking_is_symmetric_across_a_write() {
        let (_dir, svc) = service();
        let mut one = atom("The Parser reads the Header.");
        one.insert("entities".into(), json!(["Parser", "Header"]));
        let one = svc.add(one).unwrap();
        let mut two = atom("The Header comes before the Parser body.");
        two.insert("entities".into(), json!(["Parser", "Header"]));
        let two = svc.add(two).unwrap();

        let live = svc.store().current("w", None).unwrap();
        let first = live.iter().find(|a| a["id"] == one["id"]).unwrap();
        let second = live.iter().find(|a| a["id"] == two["id"]).unwrap();
        assert_eq!(second["links"], json!([one["id"].as_str().unwrap()]));
        assert_eq!(
            first["links"],
            json!([two["id"].as_str().unwrap()]),
            "the peer was rewritten, not just the newcomer"
        );
    }

    #[test]
    fn updating_a_missing_atom_says_so() {
        let (_dir, svc) = service();
        let err = svc.update("w", "nope", &Map::new()).unwrap_err();
        assert_eq!(err.to_string(), "no current atom nope");
    }

    #[test]
    fn grading_moves_the_review_clock_and_not_the_live_window() {
        let (_dir, svc) = service();
        let stored = svc.add(atom("Reviews open with a check.")).unwrap();
        let id = stored["id"].as_str().unwrap();
        let graded = svc.grade("w", id, true).unwrap();
        assert!(graded["due_at"].is_string(), "{graded:?}");
        assert_eq!(graded["review"]["reps"], json!(1));
        assert!(
            graded.get("valid_to").is_none() || graded["valid_to"].is_null(),
            "the live window is a different question"
        );
    }

    #[test]
    fn the_pack_is_one_shape_scoped_or_not() {
        let (_dir, svc) = service();
        svc.add(atom("Reviews open with a check.")).unwrap();
        let plain = svc.pack("w", None).unwrap();
        for key in ["workspace", "user", "memory", "atoms"] {
            assert!(plain.get(key).is_some(), "{key} missing from {plain}");
        }
        assert!(plain.get("set").is_none());

        let scoped = svc.pack("w", Some("Review")).unwrap();
        assert_eq!(scoped["set"], json!("review"), "the name is normalized");
        for key in ["workspace", "user", "memory", "atoms", "instructions"] {
            assert!(scoped.get(key).is_some(), "{key} missing from {scoped}");
        }
        assert!(svc.pack("w", Some("../etc")).is_err());
    }

    #[test]
    fn a_pin_round_trips_and_clears() {
        let (_dir, svc) = service();
        assert_eq!(svc.pin("w"), "");
        assert_eq!(svc.set_pin("w", "Review").unwrap(), "review");
        assert_eq!(svc.pin("w"), "review");
        assert_eq!(svc.set_pin("w", "").unwrap(), "");
        assert_eq!(svc.pin("w"), "");
        assert!(svc.set_pin("w", "../etc").is_err());
    }

    #[test]
    fn an_overflowing_card_is_archived_before_it_is_refused() {
        let (_dir, svc) = service();
        // Plain short sentences: the point is the length, and prose the
        // archive would itself refuse tests something else.
        let long = "One small claim. ".repeat(USER_CAP / 8);
        let err = svc.set_user(&long).unwrap_err();
        assert!(matches!(err, cards::WriteError::Overflow(_)), "{err}");
        // Refused, not lost: the day file has it for the miner.
        let day = clock::utcnow()[..10].to_string();
        let archived = cards::read_text(&svc.home().archive_path("global", &day));
        assert!(
            archived.contains("One small claim."),
            "the overflow was dropped rather than archived"
        );
    }

    #[test]
    fn an_attachment_is_one_shot() {
        let (_dir, svc) = service();
        svc.put_attach("w", "a log body", "build.log");
        assert_eq!(svc.peek_attach("w").unwrap().text, "a log body");
        assert_eq!(svc.peek_attach("w").unwrap().label, "build.log");
        assert_eq!(svc.take_attach("w").unwrap().text, "a log body");
        assert!(
            svc.take_attach("w").is_none(),
            "context for the next turn, not for every turn after it"
        );
    }

    #[test]
    fn an_attachment_is_capped() {
        let (_dir, svc) = service();
        let huge = "x".repeat(ATTACH_CAP + 100);
        let slot = svc.put_attach("w", &huge, "");
        assert_eq!(slot["text"].as_str().unwrap().chars().count(), ATTACH_CAP);
    }

    #[test]
    fn status_counts_by_kind_and_names_the_home() {
        let (_dir, svc) = service();
        let stored = svc.add(atom("Reviews open with a check.")).unwrap();
        svc.store()
            .delete("w", stored["id"].as_str().unwrap())
            .unwrap();
        let mut second = atom("Prefer ripgrep for search.");
        second.insert("kind".into(), json!("preference"));
        svc.add(second).unwrap();

        let status = svc.status(Some("w")).unwrap();
        assert_eq!(status["live"], json!(1));
        assert_eq!(status["tombstone"], json!(1));
        assert_eq!(status["live_by_kind"]["preference"], json!(1));
        assert_eq!(status["workspace"], json!("w"));
        assert!(status["home"].is_string());
        assert!(status["last_write_ts"].is_string());
    }
}
