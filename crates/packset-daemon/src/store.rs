//! Atoms in LMDB, in the layout the Python writer already uses.
//!
//! The key is `workspace\0id` and the value is the record as JSON. That is not
//! an internal choice: an existing seat's `memory.lmdb` has to open here and
//! read back identically, so the port is a swap rather than a migration. The
//! NUL separator is what makes a workspace scan a prefix scan, since no
//! workspace name can carry one.

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use heed::types::Bytes;
use heed::{Database, Env, EnvFlags, EnvOpenOptions};
use packset_core::record::{self, AtomError};
use serde_json::{Map, Value};

/// The map size the Python writer opens with. Growing it is compatible;
/// shrinking it below what is stored is not.
pub const MAP_SIZE: usize = 256 * 1024 * 1024;

/// One atom record.
pub type Record = Map<String, Value>;

/// One workspace's parsed live set, with the write count it was built at.
type Snapshot = (u64, Arc<Vec<Record>>);

/// The key for one atom.
#[must_use]
pub fn atom_key(workspace: &str, id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(workspace.len() + id.len() + 1);
    key.extend_from_slice(workspace.as_bytes());
    key.push(0);
    key.extend_from_slice(id.as_bytes());
    key
}

/// The prefix every key in one workspace opens with.
#[must_use]
pub fn workspace_prefix(workspace: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(workspace.len() + 1);
    key.extend_from_slice(workspace.as_bytes());
    key.push(0);
    key
}

/// The atom database, plus the lock that makes it one writer.
pub struct Store {
    env: Env,
    db: Database<Bytes, Bytes>,
    /// Held open for as long as the store is: dropping it drops the lock.
    _lock: File,
    /// Bumped by every write, so a reader can tell a stale snapshot.
    generation: AtomicU64,
    /// One parsed live set per workspace, shared by concurrent readers.
    ///
    /// Reads dominate and each one would otherwise parse the whole workspace
    /// out of the database again, so several callers asking at once pay for
    /// the same work several times. One writer means one obvious way to know
    /// a snapshot is current.
    live: RwLock<HashMap<String, Snapshot>>,
}

impl Store {
    /// Open the database under `root`, taking the single-writer lock.
    ///
    /// # Errors
    ///
    /// Fails when the home cannot be created, when another process already
    /// holds the lock, or when LMDB refuses the directory.
    pub fn open(root: &Path) -> anyhow::Result<Self> {
        fs::create_dir_all(root)?;
        let lock = take_lock(&root.join("packsetd.lock"))?;
        let db_path = root.join("memory.lmdb");
        fs::create_dir_all(&db_path)?;
        // SAFETY: LMDB maps the file; the contract is that no other process
        // writes it, which the lock above is what enforces.
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(MAP_SIZE)
                .max_dbs(1)
                .flags(EnvFlags::WRITE_MAP)
                .open(&db_path)?
        };
        let mut wtxn = env.write_txn()?;
        let db: Database<Bytes, Bytes> = env.create_database(&mut wtxn, None)?;
        wtxn.commit()?;
        Ok(Self {
            env,
            db,
            _lock: lock,
            generation: AtomicU64::new(0),
            live: RwLock::new(HashMap::new()),
        })
    }

    /// Every record in one workspace, or in all of them.
    ///
    /// # Errors
    ///
    /// Fails when the read transaction does.
    pub fn scan(&self, workspace: Option<&str>) -> anyhow::Result<Vec<Record>> {
        if workspace == Some("") {
            return Ok(Vec::new());
        }
        let rtxn = self.env.read_txn()?;
        let mut out = Vec::new();
        match workspace {
            Some(name) => {
                let prefix = workspace_prefix(name);
                for item in self.db.prefix_iter(&rtxn, &prefix)? {
                    let (_, raw) = item?;
                    push_record(&mut out, raw);
                }
            }
            None => {
                for item in self.db.iter(&rtxn)? {
                    let (_, raw) = item?;
                    push_record(&mut out, raw);
                }
            }
        }
        Ok(out)
    }

    /// One record by id, whatever its state.
    ///
    /// # Errors
    ///
    /// Fails when the read transaction does.
    pub fn get(&self, workspace: &str, id: &str) -> anyhow::Result<Option<Record>> {
        if workspace.is_empty() || id.is_empty() {
            return Ok(None);
        }
        let rtxn = self.env.read_txn()?;
        let raw = self.db.get(&rtxn, &atom_key(workspace, id))?;
        Ok(raw.and_then(|bytes| {
            serde_json::from_slice::<Value>(bytes)
                .ok()
                .and_then(|v| v.as_object().cloned())
        }))
    }

    /// Write one record, replacing whatever shared its key.
    ///
    /// # Errors
    ///
    /// Fails when the record has no workspace or id, or when the write does.
    pub fn upsert(&self, atom: &Record) -> anyhow::Result<()> {
        self.upsert_many(std::slice::from_ref(atom))
    }

    /// Write several records in one transaction.
    ///
    /// A link rewrite touches an atom and its peers together, and a reader must
    /// not be able to see one side of that.
    ///
    /// # Errors
    ///
    /// Fails when a record has no workspace or id, or when the write does.
    pub fn upsert_many(&self, atoms: &[Record]) -> anyhow::Result<()> {
        let mut wtxn = self.env.write_txn()?;
        for atom in atoms {
            let mut payload = atom.clone();
            if !matches!(payload.get("links"), Some(Value::Array(_))) {
                payload.insert("links".into(), Value::Array(Vec::new()));
            }
            let workspace = payload
                .get("workspace")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("record has no workspace"))?;
            let id = payload
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("record has no id"))?;
            let key = atom_key(workspace, id);
            let blob = serde_json::to_vec(&Value::Object(payload.clone()))?;
            self.db.put(&mut wtxn, &key, &blob)?;
        }
        wtxn.commit()?;
        // After the commit, never before: a reader that scans between a bump
        // and its write would otherwise cache the older corpus as the newer.
        self.generation.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// The live and due records in one workspace, parsed once per write.
    ///
    /// Callers that only read should take this rather than [`Store::current`]:
    /// it hands back the shared snapshot instead of a copy of it.
    ///
    /// # Errors
    ///
    /// Fails when the scan does.
    pub fn live(&self, workspace: &str) -> anyhow::Result<Arc<Vec<Record>>> {
        let generation = self.generation.load(Ordering::Acquire);
        if let Ok(cache) = self.live.read() {
            if let Some((seen, atoms)) = cache.get(workspace) {
                if *seen == generation {
                    return Ok(Arc::clone(atoms));
                }
            }
        }
        // Built outside the write lock, so a slow parse does not hold up a
        // reader whose own workspace is current.
        let now = packset_core::clock::utcnow();
        let mut atoms: Vec<Record> = self
            .scan(Some(workspace))?
            .into_iter()
            .filter(|atom| record::is_live(atom, &now) || record::is_due(atom, &now))
            .collect();
        record::filter_live_links(&mut atoms);
        let shared = Arc::new(atoms);
        // Cached only if nothing committed while the scan ran. A write that
        // landed halfway through is not in this snapshot, and storing it under
        // the newer generation would serve it as though it were: the next
        // reader would be told a committed write does not exist.
        if self.generation.load(Ordering::Acquire) == generation {
            if let Ok(mut cache) = self.live.write() {
                cache.insert(workspace.to_string(), (generation, Arc::clone(&shared)));
            }
        }
        Ok(shared)
    }

    /// The live and due records in one workspace, as a copy the caller owns.
    ///
    /// # Errors
    ///
    /// Fails when the scan does.
    pub fn current(&self, workspace: &str, set: Option<&str>) -> anyhow::Result<Vec<Record>> {
        let live = self.live(workspace)?;
        Ok(match set {
            None => live.as_ref().clone(),
            Some(name) => live
                .iter()
                .filter(|atom| atom.get("set").and_then(Value::as_str) == Some(name))
                .cloned()
                .collect(),
        })
    }

    /// Distinct workspace names with their live counts. `global` is always in.
    ///
    /// # Errors
    ///
    /// Fails when the scan does.
    pub fn workspaces(&self) -> anyhow::Result<Vec<(String, usize)>> {
        let now = packset_core::clock::utcnow();
        let mut counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for atom in self.scan(None)? {
            let Some(name) = atom.get("workspace").and_then(Value::as_str) else {
                continue;
            };
            if name.is_empty() {
                continue;
            }
            let slot = counts.entry(name.to_string()).or_insert(0);
            if record::is_live(&atom, &now) {
                *slot += 1;
            }
        }
        counts.entry("global".into()).or_insert(0);
        Ok(counts.into_iter().collect())
    }

    /// Tombstone one live record.
    ///
    /// # Errors
    ///
    /// [`AtomError`] when the id is not in the current set, else the write's.
    pub fn delete(&self, workspace: &str, id: &str) -> anyhow::Result<Record> {
        let mut tomb = self
            .current(workspace, None)?
            .into_iter()
            .find(|atom| atom.get("id").and_then(Value::as_str) == Some(id))
            .ok_or_else(|| anyhow::Error::new(AtomError(format!("no current atom {id}"))))?;
        tomb.insert("tombstone".into(), Value::Bool(true));
        tomb.insert("ts".into(), Value::String(packset_core::clock::utcnow()));
        self.upsert(&tomb)?;
        Ok(tomb)
    }
}

fn push_record(out: &mut Vec<Record>, raw: &[u8]) {
    if let Ok(Value::Object(map)) = serde_json::from_slice::<Value>(raw) {
        out.push(map);
    }
}

/// Take the exclusive lock, or say who has it.
///
/// One writer is the whole design: two processes on one `memory.lmdb` is how a
/// pack ends up with two answers to the same question.
fn take_lock(path: &Path) -> anyhow::Result<File> {
    use std::os::fd::AsRawFd;
    let file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    // SAFETY: a libc call on a fd this function owns.
    let taken = unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) };
    if taken != 0 {
        anyhow::bail!("store home is already open");
    }
    Ok(file)
}

const LOCK_EX: i32 = 2;
const LOCK_NB: i32 = 4;

extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(value: Value) -> Record {
        value.as_object().unwrap().clone()
    }

    fn store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        (dir, store)
    }

    #[test]
    fn the_key_separates_on_a_nul() {
        assert_eq!(atom_key("w", "a"), b"w\0a".to_vec());
        assert_eq!(workspace_prefix("w"), b"w\0".to_vec());
        // A workspace whose name is a prefix of another must not leak into it,
        // which is what the separator buys.
        assert!(!atom_key("wide", "a").starts_with(&workspace_prefix("w")));
    }

    #[test]
    fn a_record_round_trips() {
        let (_dir, store) = store();
        let atom = record(json!({
            "id": "one", "workspace": "w", "kind": "voice",
            "text": "A claim.", "links": ["two"], "unmodelled": {"x": 1}
        }));
        store.upsert(&atom).unwrap();
        let back = store.get("w", "one").unwrap().unwrap();
        assert_eq!(back["text"], json!("A claim."));
        assert_eq!(back["links"], json!(["two"]));
        assert_eq!(back["unmodelled"], json!({"x": 1}), "fields survive");
    }

    #[test]
    fn a_scan_is_scoped_to_one_workspace() {
        let (_dir, store) = store();
        for (ws, id) in [("w", "a"), ("w", "b"), ("wide", "c")] {
            store
                .upsert(&record(json!({"id": id, "workspace": ws, "text": id})))
                .unwrap();
        }
        let mine = store.scan(Some("w")).unwrap();
        assert_eq!(mine.len(), 2, "{mine:?}");
        assert_eq!(store.scan(Some("wide")).unwrap().len(), 1);
        assert_eq!(store.scan(None).unwrap().len(), 3);
        assert!(store.scan(Some("")).unwrap().is_empty());
    }

    #[test]
    fn current_drops_the_expired_and_keeps_the_due() {
        let (_dir, store) = store();
        store
            .upsert(&record(
                json!({"id": "live", "workspace": "w", "text": "a"}),
            ))
            .unwrap();
        store
            .upsert(&record(json!({
                "id": "gone", "workspace": "w", "text": "b",
                "valid_to": "2000-01-01T00:00:00.000Z"
            })))
            .unwrap();
        // Expired for the live set but still on the review clock, which is a
        // different question and keeps it in reach.
        store
            .upsert(&record(json!({
                "id": "due", "workspace": "w", "text": "c",
                "valid_to": "2000-01-01T00:00:00.000Z",
                "due_at": "2000-01-01T00:00:00.000Z"
            })))
            .unwrap();
        let ids: Vec<String> = store
            .current("w", None)
            .unwrap()
            .iter()
            .map(|a| a["id"].as_str().unwrap().to_string())
            .collect();
        assert!(ids.contains(&"live".to_string()), "{ids:?}");
        assert!(ids.contains(&"due".to_string()), "{ids:?}");
        assert!(!ids.contains(&"gone".to_string()), "{ids:?}");
    }

    #[test]
    fn current_narrows_links_to_what_it_returned() {
        let (_dir, store) = store();
        store
            .upsert(&record(json!({
                "id": "a", "workspace": "w", "text": "a", "links": ["b", "gone"]
            })))
            .unwrap();
        store
            .upsert(&record(json!({"id": "b", "workspace": "w", "text": "b"})))
            .unwrap();
        let live = store.current("w", None).unwrap();
        let a = live.iter().find(|x| x["id"] == json!("a")).unwrap();
        assert_eq!(a["links"], json!(["b"]), "a dangling link is not returned");
    }

    #[test]
    fn a_set_scope_filters_the_current_view() {
        let (_dir, store) = store();
        store
            .upsert(&record(
                json!({"id": "a", "workspace": "w", "text": "a", "set": "review"}),
            ))
            .unwrap();
        store
            .upsert(&record(json!({"id": "b", "workspace": "w", "text": "b"})))
            .unwrap();
        assert_eq!(store.current("w", Some("review")).unwrap().len(), 1);
        assert_eq!(store.current("w", None).unwrap().len(), 2);
    }

    #[test]
    fn workspaces_count_the_live_and_always_name_global() {
        let (_dir, store) = store();
        store
            .upsert(&record(json!({"id": "a", "workspace": "w", "text": "a"})))
            .unwrap();
        store
            .upsert(&record(json!({
                "id": "b", "workspace": "w", "text": "b", "tombstone": true
            })))
            .unwrap();
        let found = store.workspaces().unwrap();
        assert!(found.contains(&("w".to_string(), 1)), "{found:?}");
        assert!(
            found.iter().any(|(name, _)| name == "global"),
            "an empty seat still has somewhere to write: {found:?}"
        );
    }

    #[test]
    fn deleting_leaves_a_tombstone_rather_than_a_hole() {
        let (_dir, store) = store();
        store
            .upsert(&record(json!({"id": "a", "workspace": "w", "text": "a"})))
            .unwrap();
        let tomb = store.delete("w", "a").unwrap();
        assert_eq!(tomb["tombstone"], json!(true));
        // The record is still there to be read; it has left the live set.
        assert!(store.get("w", "a").unwrap().is_some());
        assert!(store.current("w", None).unwrap().is_empty());
        assert!(store.delete("w", "a").is_err(), "twice is not current");
    }

    #[test]
    fn a_second_writer_is_refused_the_home() {
        let dir = tempfile::tempdir().unwrap();
        let _first = Store::open(dir.path()).unwrap();
        let second = Store::open(dir.path());
        assert!(second.is_err(), "one writer is the whole design");
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use serde_json::json;

    fn record(value: Value) -> Record {
        value.as_object().unwrap().clone()
    }

    fn store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        (dir, store)
    }

    #[test]
    fn a_write_is_visible_to_the_next_read() {
        let (_dir, store) = store();
        assert!(store.live("w").unwrap().is_empty());
        store
            .upsert(&record(json!({"id": "a", "workspace": "w", "text": "a"})))
            .unwrap();
        assert_eq!(store.live("w").unwrap().len(), 1, "the snapshot went stale");
        store
            .upsert(&record(json!({"id": "b", "workspace": "w", "text": "b"})))
            .unwrap();
        assert_eq!(store.live("w").unwrap().len(), 2);
    }

    #[test]
    fn a_repeated_read_hands_back_the_same_snapshot() {
        let (_dir, store) = store();
        store
            .upsert(&record(json!({"id": "a", "workspace": "w", "text": "a"})))
            .unwrap();
        let first = store.live("w").unwrap();
        let second = store.live("w").unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "two readers should share one parse"
        );
        store
            .upsert(&record(json!({"id": "b", "workspace": "w", "text": "b"})))
            .unwrap();
        let third = store.live("w").unwrap();
        assert!(!Arc::ptr_eq(&first, &third), "a write invalidates it");
    }

    #[test]
    fn a_tombstone_leaves_the_snapshot() {
        let (_dir, store) = store();
        store
            .upsert(&record(json!({"id": "a", "workspace": "w", "text": "a"})))
            .unwrap();
        assert_eq!(store.live("w").unwrap().len(), 1);
        store.delete("w", "a").unwrap();
        assert!(
            store.live("w").unwrap().is_empty(),
            "a delete must invalidate too"
        );
    }

    #[test]
    fn one_workspace_write_does_not_serve_another_stale() {
        let (_dir, store) = store();
        store
            .upsert(&record(json!({"id": "a", "workspace": "one", "text": "a"})))
            .unwrap();
        assert_eq!(store.live("one").unwrap().len(), 1);
        assert!(store.live("two").unwrap().is_empty());
        store
            .upsert(&record(json!({"id": "b", "workspace": "two", "text": "b"})))
            .unwrap();
        assert_eq!(store.live("two").unwrap().len(), 1);
        assert_eq!(store.live("one").unwrap().len(), 1, "still correct");
    }

    #[test]
    fn readers_racing_a_writer_never_see_a_snapshot_that_skips_a_write() {
        // The generation is read before the scan and bumped before the write,
        // so a snapshot built across a write is stale rather than labelled as
        // including it. What must never happen is a later read seeing fewer
        // atoms than an earlier one.
        let (dir, store) = store();
        let store = Arc::new(store);
        let _ = dir;
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let writer = {
            let store = Arc::clone(&store);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                for i in 0..200 {
                    store
                        .upsert(&record(json!({
                            "id": format!("a{i}"), "workspace": "w", "text": "x"
                        })))
                        .unwrap();
                }
                stop.store(true, std::sync::atomic::Ordering::Release);
            })
        };

        let readers: Vec<_> = (0..4)
            .map(|_| {
                let store = Arc::clone(&store);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut high = 0usize;
                    while !stop.load(std::sync::atomic::Ordering::Acquire) {
                        let seen = store.live("w").unwrap().len();
                        assert!(seen >= high, "went backwards: {seen} after {high}");
                        high = seen;
                    }
                })
            })
            .collect();

        writer.join().unwrap();
        for reader in readers {
            reader.join().unwrap();
        }
        assert_eq!(store.live("w").unwrap().len(), 200, "every write landed");
    }
}
