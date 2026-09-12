//! Loopback HTTP client for packsetd.
//!
//! Reads `PACKSET_URL` or `INSIDE_MEMORY_URL`. search/get against packsetd; no SQLite.
//! Does not open LMDB.

use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(5);

fn path_seg(id: &str) -> String {
    let mut out = String::with_capacity(id.len());
    for b in id.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("packset url missing")]
    NoUrl,
    #[error("http: {0}")]
    Http(#[from] Box<ureq::Error>),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("bad response: {0}")]
    Bad(String),
}

#[derive(Debug, Clone)]
pub struct PacksetClient {
    base: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    pub id: Option<String>,
    pub text: String,
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub kind: String,
}

/// A refusal, carrying the reason the writer gave in its body.
fn refused(url: &str, e: ureq::Error) -> Error {
    match e {
        ureq::Error::Status(code, response) => {
            let text = response.into_string().unwrap_or_default();
            let reason = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v.get("error").and_then(|r| r.as_str()).map(str::to_string))
                .unwrap_or(text);
            let reason = reason.trim();
            if reason.is_empty() {
                Error::Bad(format!("{url}: status code {code}"))
            } else {
                Error::Bad(format!("{url}: {code}: {reason}"))
            }
        }
        other => Error::Http(Box::new(other)),
    }
}

impl PacksetClient {
    pub fn new(base: impl Into<String>) -> Self {
        let mut base = base.into();
        while base.ends_with('/') {
            base.pop();
        }
        Self { base }
    }

    pub fn from_env() -> Result<Self, Error> {
        let url = env::var("PACKSET_URL")
            .or_else(|_| env::var("INSIDE_MEMORY_URL"))
            .map_err(|_| Error::NoUrl)?;
        if url.is_empty() || url == "off" {
            return Err(Error::NoUrl);
        }
        Ok(Self::new(url))
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn workspace(&self) -> String {
        if let Ok(w) = env::var("PACKSET_WORKSPACE") {
            if !w.is_empty() {
                return w;
            }
        }
        let cwd = env::var("GROKOS_WORKSPACE")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from)
            .or_else(|| env::current_dir().ok())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        self.workspace_for_cwd(&cwd)
    }

    /// Workspace id from `/v1/identity` for `cwd`, or `dir:<abs>` if that call fails.
    pub fn workspace_for_cwd(&self, cwd: &std::path::Path) -> String {
        let abs = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
        let url = format!("{}/v1/identity", self.base);
        let body = ureq::get(&url)
            .query("cwd", abs.to_string_lossy().as_ref())
            .timeout(TIMEOUT)
            .call()
            .ok()
            .and_then(|r| r.into_string().ok());
        if let Some(body) = body {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&body) {
                if let Some(ws) = val.get("workspace").and_then(|v| v.as_str()) {
                    if !ws.is_empty() {
                        return ws.to_string();
                    }
                }
            }
        }
        format!("dir:{}", abs.display())
    }

    pub fn health(&self) -> Result<String, Error> {
        let url = format!("{}/health", self.base);
        let body = ureq::get(&url)
            .timeout(TIMEOUT)
            .call()
            .map_err(|e| refused(&url, e))?
            .into_string()?;
        Ok(body)
    }

    pub fn get_atom(&self, workspace: &str, id: &str) -> Result<serde_json::Value, Error> {
        let encoded = path_seg(id);
        let url = format!("{}/v1/atoms/{encoded}", self.base);
        let resp = match ureq::get(&url)
            .query("workspace", workspace)
            .timeout(TIMEOUT)
            .call()
        {
            Ok(resp) => resp,
            Err(ureq::Error::Status(404, _)) => {
                return Err(Error::Bad(format!("no atom {id}")));
            }
            Err(e) => return Err(Error::Http(Box::new(e))),
        };
        Ok(resp.into_json()?)
    }

    pub fn list_atoms(&self, workspace: &str) -> Result<Vec<serde_json::Value>, Error> {
        self.atoms_as_of(workspace, None)
    }

    /// Live-now atoms, or the ones that were live at `as_of`.
    ///
    /// # Errors
    ///
    /// The request's, or a body that is not JSON.
    pub fn atoms_as_of(
        &self,
        workspace: &str,
        as_of: Option<&str>,
    ) -> Result<Vec<serde_json::Value>, Error> {
        let url = format!("{}/v1/atoms", self.base);
        let mut req = ureq::get(&url)
            .query("workspace", workspace)
            .timeout(TIMEOUT);
        if let Some(at) = as_of {
            req = req.query("as_of", at);
        }
        let body: serde_json::Value = req.call().map_err(|e| refused(&url, e))?.into_json()?;
        let atoms = body
            .get("atoms")
            .cloned()
            .unwrap_or(serde_json::Value::Array(vec![]));
        Ok(serde_json::from_value(atoms)?)
    }

    pub fn search(&self, workspace: &str, q: &str, limit: u32) -> Result<Vec<Hit>, Error> {
        self.search_as_of(workspace, q, limit, None)
    }

    /// Ranked hits, optionally over the atoms that were live at `as_of`.
    ///
    /// # Errors
    ///
    /// The request's, or a body that is not JSON.
    pub fn search_as_of(
        &self,
        workspace: &str,
        q: &str,
        limit: u32,
        as_of: Option<&str>,
    ) -> Result<Vec<Hit>, Error> {
        self.search_opts(workspace, q, limit, as_of, false)
    }

    /// Ranked hits, optionally dated and optionally through the measured
    /// cross-encoder stage.
    ///
    /// Off by default. On, the writer spends a forward pass per candidate and
    /// the request waits for that rather than the usual five-second budget.
    ///
    /// # Errors
    ///
    /// The request's, or a body that is not a hit list.
    pub fn search_opts(
        &self,
        workspace: &str,
        q: &str,
        limit: u32,
        as_of: Option<&str>,
        rerank: bool,
    ) -> Result<Vec<Hit>, Error> {
        let url = format!("{}/v1/search", self.base);
        let timeout = if rerank {
            Duration::from_secs(60)
        } else {
            TIMEOUT
        };
        let mut req = ureq::get(&url)
            .query("workspace", workspace)
            .query("q", q)
            .query("limit", &limit.to_string())
            .timeout(timeout);
        if let Some(at) = as_of {
            req = req.query("as_of", at);
        }
        if rerank {
            req = req.query("rerank", "1");
        }
        let body: serde_json::Value = req.call().map_err(|e| refused(&url, e))?.into_json()?;
        let hits = body
            .get("hits")
            .cloned()
            .unwrap_or(serde_json::Value::Array(vec![]));
        Ok(serde_json::from_value(hits)?)
    }

    /// Seat home, atom counts by kind, pin, index and embedder.
    ///
    /// # Errors
    ///
    /// The request's, or a body that is not JSON.
    pub fn status(&self, workspace: Option<&str>) -> Result<serde_json::Value, Error> {
        let url = format!("{}/v1/status", self.base);
        let mut req = ureq::get(&url).timeout(TIMEOUT);
        if let Some(workspace) = workspace {
            req = req.query("workspace", workspace);
        }
        Ok(req.call().map_err(|e| refused(&url, e))?.into_json()?)
    }

    /// The set a workspace is pinned to.
    ///
    /// # Errors
    ///
    /// The request's, or a body that is not JSON.
    pub fn pin(&self, workspace: &str) -> Result<serde_json::Value, Error> {
        let url = format!("{}/v1/pin", self.base);
        Ok(ureq::get(&url)
            .query("workspace", workspace)
            .timeout(TIMEOUT)
            .call()
            .map_err(|e| refused(&url, e))?
            .into_json()?)
    }

    /// Pin a workspace to a set.
    ///
    /// # Errors
    ///
    /// The request's, or a body that is not JSON.
    pub fn set_pin(&self, workspace: &str, name: &str) -> Result<serde_json::Value, Error> {
        let url = format!("{}/v1/pin", self.base);
        Ok(ureq::put(&url)
            .timeout(TIMEOUT)
            .send_json(serde_json::json!({ "workspace": workspace, "name": name }))
            .map_err(|e| refused(&url, e))?
            .into_json()?)
    }

    /// The deed accessions a workspace's live atoms cite, sorted.
    ///
    /// The accession is the only identifier crossing the tracker, the pack and
    /// the deed store, so this is what `deedar evidence -` reads.
    ///
    /// # Errors
    ///
    /// The request's, or a body that is not JSON.
    pub fn accessions(&self, workspace: &str) -> Result<Vec<String>, Error> {
        let url = format!("{}/v1/accessions", self.base);
        let body: serde_json::Value = ureq::get(&url)
            .query("workspace", workspace)
            .timeout(TIMEOUT)
            .call()
            .map_err(|e| refused(&url, e))?
            .into_json()?;
        let found = body
            .get("accessions")
            .cloned()
            .unwrap_or(serde_json::Value::Array(vec![]));
        Ok(serde_json::from_value(found)?)
    }
    /// Every live atom in a workspace.
    ///
    /// The bodies, not the join keys: this is what a handover carries when
    /// somebody is given what the seat learned rather than only what it cites.
    ///
    /// # Errors
    ///
    /// The request's, or a body that is not JSON.
    pub fn atoms(&self, workspace: &str) -> Result<Vec<serde_json::Value>, Error> {
        self.atoms_as_of(workspace, None)
    }

    /// The live atoms in a workspace that cite one deed accession.
    ///
    /// # Errors
    ///
    /// The request's, or a body that is not JSON.
    pub fn citers(
        &self,
        workspace: &str,
        accession: &str,
    ) -> Result<Vec<serde_json::Value>, Error> {
        let url = format!("{}/v1/citers", self.base);
        let body: serde_json::Value = ureq::get(&url)
            .query("workspace", workspace)
            .query("accession", accession)
            .timeout(TIMEOUT)
            .call()
            .map_err(|e| refused(&url, e))?
            .into_json()?;
        let found = body
            .get("atoms")
            .cloned()
            .unwrap_or(serde_json::Value::Array(vec![]));
        Ok(serde_json::from_value(found)?)
    }

    pub fn post_atom(&self, atom: &serde_json::Value) -> Result<serde_json::Value, Error> {
        let url = format!("{}/v1/atoms", self.base);
        let body: serde_json::Value = ureq::post(&url)
            .timeout(TIMEOUT)
            .send_json(atom.clone())
            .map_err(|e| refused(&url, e))?
            .into_json()?;
        Ok(body)
    }
}
