//! The `/v1` surface.
//!
//! Loopback only, and `127.0.0.1` rather than `localhost`: the name resolves
//! to whatever the resolver says, which on a dual-stack seat is not always the
//! interface the writer bound. The address is the contract.
//!
//! This layer decodes and encodes and nothing else. What a verb means lives in
//! [`crate::service`].

use std::collections::HashMap;
use std::sync::Arc;

use packset_core::record::AtomError;
use serde_json::{json, Map, Value};
use tiny_http::{Header, Method, Request, Response, Server};

use crate::cards::WriteError;
use crate::service::Service;

/// The address the writer will bind, and no other.
pub const LOOPBACK: &str = "127.0.0.1";
/// The port the clients look for.
pub const DEFAULT_PORT: u16 = 8761;

/// What a route decided to answer.
struct Answer {
    code: u16,
    body: Value,
}

impl Answer {
    fn ok(body: Value) -> Self {
        Self { code: 200, body }
    }

    fn err(code: u16, message: impl std::fmt::Display) -> Self {
        Self {
            code,
            body: json!({ "error": message.to_string() }),
        }
    }
}

/// Serve until the process is stopped.
///
/// # Errors
///
/// Fails when the address cannot be bound.
pub fn serve(service: Arc<Service>, host: &str, port: u16) -> anyhow::Result<()> {
    if host != LOOPBACK {
        anyhow::bail!("packsetd listens on {LOOPBACK} only");
    }
    let server = Server::http((host, port))
        .map_err(|e| anyhow::anyhow!("cannot bind {host}:{port}: {e}"))?;
    eprintln!("packsetd: listening on http://{host}:{port}");
    for request in server.incoming_requests() {
        let service = Arc::clone(&service);
        // A thread per request, the way the writer being replaced does it: the
        // store's own lock is what serialises the writes.
        std::thread::spawn(move || handle(&service, request));
    }
    Ok(())
}

fn handle(service: &Service, mut request: Request) {
    let url = request.url().to_string();
    let (path, query) = split_query(&url);
    let method = request.method().clone();

    if method == Method::Get && path == "/health" {
        let response = Response::from_string("packsetd ok").with_header(text_plain());
        let _ = request.respond(response);
        return;
    }

    let body = if matches!(method, Method::Post | Method::Put) {
        match read_json(&mut request) {
            Ok(map) => map,
            Err(message) => {
                respond(request, &Answer::err(400, message));
                return;
            }
        }
    } else {
        Map::new()
    };

    let answer = route(service, &method, path, &query, &body);
    respond(request, &answer);
}

fn route(
    service: &Service,
    method: &Method,
    path: &str,
    query: &HashMap<String, String>,
    body: &Map<String, Value>,
) -> Answer {
    match (method, path) {
        // The old name answered here once; a client still asking for it is
        // reading a store this writer does not serve.
        (Method::Get, "/__inside_memd/health") => Answer::err(404, "not found"),
        (Method::Get, "/v1/status") => {
            let workspace = query.get("workspace").filter(|w| !w.is_empty());
            answer(service.status(workspace.map(String::as_str)))
        }
        (Method::Get, "/v1/workspaces") => match service.store().workspaces() {
            Ok(found) => Answer::ok(json!({
                "workspaces": found
                    .into_iter()
                    .map(|(name, live)| json!({"name": name, "live": live}))
                    .collect::<Vec<_>>()
            })),
            Err(e) => Answer::err(400, e),
        },
        (Method::Get, "/v1/pin") => match required(query, "workspace") {
            Err(a) => a,
            Ok(workspace) => answer(service.pin_payload(&workspace)),
        },
        (Method::Get, "/v1/pack") => match required(query, "workspace") {
            Err(a) => a,
            Ok(workspace) => {
                let set = query.get("set").filter(|s| !s.is_empty());
                answer(service.pack(&workspace, set.map(String::as_str)))
            }
        },
        (Method::Get, "/v1/set") => match required(query, "workspace") {
            Err(a) => a,
            Ok(workspace) => match required(query, "name") {
                Err(a) => a,
                Ok(name) => answer(service.pack(&workspace, Some(&name))),
            },
        },
        (Method::Get, "/v1/atoms") => match required(query, "workspace") {
            Err(a) => a,
            Ok(workspace) => match service.store().current(&workspace, None) {
                Ok(atoms) => Answer::ok(json!({ "atoms": atoms })),
                Err(e) => Answer::err(400, e),
            },
        },
        (Method::Get, _) if path.starts_with("/v1/atoms/") => {
            let id = &path["/v1/atoms/".len()..];
            if id.is_empty() || id.contains('/') {
                return Answer::err(404, "not found");
            }
            match required(query, "workspace") {
                Err(a) => a,
                Ok(workspace) => match service.store().get(&workspace, id) {
                    Err(e) => Answer::err(400, e),
                    Ok(None) => Answer::err(404, "no atom"),
                    Ok(Some(atom)) => {
                        let now = packset_core::clock::utcnow();
                        if packset_core::record::is_live(&atom, &now) {
                            Answer::ok(Value::Object(atom))
                        } else {
                            Answer::err(404, "no atom")
                        }
                    }
                },
            }
        }
        (Method::Get, "/v1/identity") => {
            let cwd = query.get("cwd").cloned().unwrap_or_else(|| ".".into());
            let harness = query
                .get("harness")
                .filter(|h| !h.is_empty())
                .cloned()
                .unwrap_or_else(|| "any".into());
            match crate::workspace::identity(
                std::path::Path::new(&cwd),
                packset_core::identity::Strategy::PerRepo,
                &harness,
                None,
                None,
                0,
                None,
            ) {
                Ok(value) => Answer::ok(value),
                Err(message) => Answer::err(400, message),
            }
        }
        (Method::Get, "/v1/rules") => {
            let cwd = query.get("cwd").cloned().unwrap_or_else(|| ".".into());
            let with_body = truthy(query.get("body").map(String::as_str));
            Answer::ok(crate::context::rules_payload(
                std::path::Path::new(&cwd),
                &service.home().user_path(),
                with_body,
            ))
        }
        (Method::Get, "/v1/skills") => {
            let cwd = query.get("cwd").cloned().unwrap_or_else(|| ".".into());
            let name = query.get("name").filter(|n| !n.is_empty());
            // Global skills live under the seat's own home, not the pack home:
            // a pack can be moved between seats and a skill catalog cannot.
            let home = std::env::var_os("HOME")
                .map_or_else(|| std::path::PathBuf::from("."), std::path::PathBuf::from);
            Answer::ok(crate::context::skills_payload(
                std::path::Path::new(&cwd),
                &home,
                name.map(String::as_str),
            ))
        }
        (Method::Get, "/v1/map") => {
            let cwd = query.get("cwd").cloned().unwrap_or_else(|| ".".into());
            Answer::ok(crate::context::repo_map(std::path::Path::new(&cwd)))
        }
        (Method::Get, "/v1/search") => match required(query, "workspace") {
            Err(a) => a,
            Ok(workspace) => {
                let limit = match query.get("limit").filter(|l| !l.is_empty()) {
                    None => 16usize,
                    Some(raw) => match raw.parse::<i64>() {
                        Ok(v) => v.max(0) as usize,
                        Err(_) => return Answer::err(400, "limit must be an integer"),
                    },
                };
                let q = query.get("q").cloned().unwrap_or_default();
                let requested = query.get("set").filter(|s| !s.is_empty());
                // A named set scopes the atoms and swaps the prose for that
                // set's cards, but the atom list stays the whole live set: the
                // scope is a filter in the scorer, not a smaller corpus.
                let (set, user, memory) = match requested {
                    Some(raw) => match packset_core::set_name::check(raw) {
                        Err(e) => return Answer::err(400, e),
                        Ok(named) => {
                            let home = service.home();
                            (
                                Some(named.clone()),
                                crate::cards::read_text(&home.set_user_path(&workspace, &named)),
                                crate::cards::read_text(&home.set_memory_path(&workspace, &named)),
                            )
                        }
                    },
                    None => (
                        None,
                        crate::cards::read_text(&service.home().user_path()),
                        crate::cards::read_text(&service.home().memory_path(&workspace)),
                    ),
                };
                match service.store().current(&workspace, None) {
                    Err(e) => Answer::err(400, e),
                    Ok(atoms) => {
                        let now = packset_core::clock::utcnow();
                        let set = set.as_deref();
                        let ranked = packset_core::search::search_linear(
                            &user, &memory, &atoms, &q, limit, set, &now,
                        );
                        let due = packset_core::search::due_hits(&atoms, set, &now);
                        let hits = if due.is_empty() {
                            ranked
                        } else {
                            packset_core::search::front_due(due, ranked, limit)
                        };
                        Answer::ok(json!({"hits": hits, "engine": "linear"}))
                    }
                }
            }
        },
        (Method::Get, "/v1/recall") => match required(query, "workspace") {
            Err(a) => a,
            Ok(workspace) => {
                let limit = match query.get("limit").filter(|l| !l.is_empty()) {
                    None => None,
                    Some(raw) => match raw.parse::<i64>() {
                        Ok(v) => Some(v),
                        Err(_) => return Answer::err(400, "limit must be an integer"),
                    },
                };
                let seeds: Vec<String> = query
                    .get("seed")
                    .map(|raw| {
                        raw.split(',')
                            .filter(|s| !s.is_empty())
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                let hints = packset_core::recall::Hints {
                    text: query.get("q").cloned().unwrap_or_default(),
                    entities: Vec::new(),
                };
                match service.store().current(&workspace, None) {
                    Err(e) => Answer::err(400, e),
                    Ok(atoms) => Answer::ok(json!({
                        "atoms": packset_core::recall::recall(
                            &atoms,
                            &seeds,
                            &hints,
                            limit,
                            &packset_core::clock::utcnow(),
                        )
                    })),
                }
            }
        },
        (Method::Get, "/v1/attach") => match required(query, "workspace") {
            Err(a) => a,
            Ok(workspace) => {
                let peek = truthy(query.get("peek").map(String::as_str));
                let slot = if peek {
                    service.peek_attach(&workspace)
                } else {
                    service.take_attach(&workspace)
                };
                let slot = slot.unwrap_or_default();
                Answer::ok(json!({
                    "workspace": workspace,
                    "text": slot.text,
                    "label": slot.label,
                }))
            }
        },
        (Method::Put, "/v1/pin") => match required(body, "workspace") {
            Err(a) => a,
            Ok(workspace) => {
                let name = body.get("set").and_then(Value::as_str).unwrap_or("");
                match service.set_pin(&workspace, name) {
                    Ok(pinned) => Answer::ok(json!({"workspace": workspace, "set": pinned})),
                    Err(e) => Answer::err(400, e),
                }
            }
        },
        (Method::Put, "/v1/user") => {
            let text = body.get("text").and_then(Value::as_str).unwrap_or("");
            card_answer(service.set_user(text))
        }
        (Method::Put, "/v1/memory") => {
            // No workspace required, which is the writer being replaced: an
            // empty name slugs to `workspace` and the card lands there rather
            // than being refused. Odd, and load-bearing for anything already
            // sending one.
            let workspace = body.get("workspace").and_then(Value::as_str).unwrap_or("");
            let text = body.get("text").and_then(Value::as_str).unwrap_or("");
            card_answer(service.set_memory(workspace, text))
        }
        (Method::Post, "/v1/atoms") => answer(service.add(body.clone())),
        (Method::Post, "/v1/atoms/update") => {
            let (Some(workspace), Some(id)) = (
                body.get("workspace").and_then(Value::as_str),
                body.get("id").and_then(Value::as_str),
            ) else {
                return Answer::err(400, "workspace and id required");
            };
            let empty = Map::new();
            let fields = body
                .get("fields")
                .and_then(Value::as_object)
                .unwrap_or(&empty);
            answer(service.update(workspace, id, fields))
        }
        (Method::Post, "/v1/atoms/delete") => {
            let (Some(workspace), Some(id)) = (
                body.get("workspace").and_then(Value::as_str),
                body.get("id").and_then(Value::as_str),
            ) else {
                return Answer::err(400, "workspace and id required");
            };
            answer(service.store().delete(workspace, id).map(Value::Object))
        }
        (Method::Post, "/v1/grade") => {
            let workspace = body.get("workspace").and_then(Value::as_str).unwrap_or("");
            let id = body.get("id").and_then(Value::as_str).unwrap_or("");
            if workspace.is_empty() || id.is_empty() {
                return Answer::err(400, "workspace and id required");
            }
            let recalled = match body.get("recalled") {
                None | Some(Value::Null) => true,
                Some(Value::Bool(b)) => *b,
                Some(Value::String(s)) => {
                    !matches!(s.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no")
                }
                Some(other) => other.as_i64().unwrap_or(1) != 0,
            };
            answer(service.grade(workspace, id, recalled))
        }
        (Method::Post, "/v1/attach") => match required(body, "workspace") {
            Err(a) => a,
            Ok(workspace) => {
                let text = match body.get("text") {
                    Some(Value::String(s)) => s.clone(),
                    // No text at all means the body names a file to read, so a
                    // client can hand over a log without carrying it.
                    Some(Value::Null) | None => crate::context::read_attach_source(
                        body.get("path").and_then(Value::as_str).unwrap_or(""),
                        crate::context::ATTACH_CAP,
                    ),
                    Some(other) => other.to_string(),
                };
                let label = body.get("label").and_then(Value::as_str).unwrap_or("");
                Answer::ok(service.put_attach(&workspace, &text, label))
            }
        },
        _ => Answer::err(404, "not found"),
    }
}

/// Turn a service result into an answer, keeping the store's own message.
fn answer<T: Into<Value>>(result: anyhow::Result<T>) -> Answer {
    match result {
        Ok(value) => Answer::ok(value.into()),
        Err(e) => Answer::err(400, root_message(&e)),
    }
}

/// The innermost message, which is the one a client can act on.
fn root_message(err: &anyhow::Error) -> String {
    if let Some(atom) = err.downcast_ref::<AtomError>() {
        return atom.0.clone();
    }
    err.to_string()
}

/// Overflow answers 413, because the client can shorten and retry; anything
/// else about a card is a refusal it has to fix.
fn card_answer(result: Result<(), WriteError>) -> Answer {
    match result {
        Ok(()) => Answer::ok(json!({"ok": true})),
        Err(WriteError::Overflow(o)) => Answer::err(413, o),
        Err(other) => Answer::err(400, other),
    }
}

trait Lookup {
    fn lookup(&self, key: &str) -> Option<String>;
}

impl Lookup for HashMap<String, String> {
    fn lookup(&self, key: &str) -> Option<String> {
        self.get(key).filter(|v| !v.is_empty()).cloned()
    }
}

impl Lookup for Map<String, Value> {
    fn lookup(&self, key: &str) -> Option<String> {
        self.get(key)
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    }
}

fn required<L: Lookup>(source: &L, key: &str) -> Result<String, Answer> {
    source
        .lookup(key)
        .ok_or_else(|| Answer::err(400, format!("{key} required")))
}

fn truthy(raw: Option<&str>) -> bool {
    matches!(raw, Some("1" | "true" | "yes"))
}

fn read_json(request: &mut Request) -> Result<Map<String, Value>, String> {
    let mut raw = String::new();
    request
        .as_reader()
        .read_to_string(&mut raw)
        .map_err(|e| e.to_string())?;
    if raw.trim().is_empty() {
        return Ok(Map::new());
    }
    let value: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| "JSON object required".to_string())
}

fn split_query(url: &str) -> (&str, HashMap<String, String>) {
    let Some((path, raw)) = url.split_once('?') else {
        return (url, HashMap::new());
    };
    let mut out = HashMap::new();
    for pair in raw.split('&').filter(|p| !p.is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        // First wins, matching a parse that takes element zero of the list.
        out.entry(percent_decode(key))
            .or_insert_with(|| percent_decode(value));
    }
    (path, out)
}

fn percent_decode(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(byte) => {
                        out.push(byte);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn text_plain() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"text/plain"[..]).expect("static header")
}

fn application_json() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).expect("static header")
}

fn respond(request: Request, answer: &Answer) {
    let body = serde_json::to_string(&answer.body).unwrap_or_else(|_| "{}".into());
    let response = Response::from_string(body)
        .with_status_code(answer.code)
        .with_header(application_json());
    let _ = request.respond(response);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_query_splits_and_decodes() {
        let (path, query) = split_query("/v1/pack?workspace=git%3Agithub.com%2FHaoZeke%2Fvissue");
        assert_eq!(path, "/v1/pack");
        assert_eq!(
            query.get("workspace").map(String::as_str),
            Some("git:github.com/HaoZeke/vissue")
        );
    }

    #[test]
    fn a_path_with_no_query_is_left_alone() {
        let (path, query) = split_query("/v1/workspaces");
        assert_eq!(path, "/v1/workspaces");
        assert!(query.is_empty());
    }

    #[test]
    fn the_first_value_of_a_repeated_key_wins() {
        let (_, query) = split_query("/v1/pack?workspace=a&workspace=b");
        assert_eq!(query.get("workspace").map(String::as_str), Some("a"));
    }

    #[test]
    fn a_flag_reads_the_three_spellings_and_nothing_else() {
        assert!(truthy(Some("1")));
        assert!(truthy(Some("true")));
        assert!(truthy(Some("yes")));
        assert!(!truthy(Some("on")));
        assert!(!truthy(Some("")));
        assert!(!truthy(None));
    }

    #[test]
    fn a_plus_is_a_space_and_a_stray_percent_survives() {
        assert_eq!(percent_decode("a+b"), "a b");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz"), "%zz");
    }
}
