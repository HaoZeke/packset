//! `packset`: start, stop and inspect the pack writer.
//!
//! The daemon owns the store and every client speaks HTTP to it, so a seat
//! needs one command that answers "is it up, and at what URL". Everything here
//! is either that question or a thin read of `/v1`.
//!
//! ```console
//! packset ensure | start | stop | status | port | url | which
//! packset pin [NAME]
//! packset accessions [WORKSPACE]
//! packset citers ACCESSION [WORKSPACE]
//! ```
//!
//! Clients export `PACKSET_URL`; `INSIDE_MEMORY_URL` is an alias.

mod procfs;

use std::env;
use std::fs::{self, OpenOptions};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use packset_client::PacksetClient;

/// The port a seat uses when it says nothing.
const DEFAULT_PORT: u16 = 8761;

/// How long `start` waits for the daemon to bind.
const STARTUP: Duration = Duration::from_secs(5);

/// What `/health` says when the answer came from our own writer.
const OURS: &[&str] = &["packsetd", "inside-memd"];

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("packset: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let (verb, rest) = args
        .split_first()
        .map_or(("ensure", &[][..]), |(head, tail)| (head.as_str(), tail));
    let port = port();

    match verb {
        "ensure" => {
            if procfs::listening(port) && !is_ours(port) {
                anyhow::bail!("port {port} is held by something else; set PACKSET_PORT");
            }
            if !procfs::listening(port) {
                start(port)?;
            }
            print_url(port);
            Ok(())
        }
        "start" => start(port),
        "stop" => stop(port),
        "status" => status(port, rest.first().map(String::as_str)),
        "port" => {
            println!("{port}");
            Ok(())
        }
        "url" => {
            print_url(port);
            Ok(())
        }
        "which" => {
            let daemon = resolve_daemon()
                .ok_or_else(|| anyhow::anyhow!("no packsetd binary; cargo build --release"))?;
            println!("{}", daemon.display());
            Ok(())
        }
        "pin" => pin(port, rest.first().map(String::as_str)),
        "accessions" => accessions(port, rest.first().map(String::as_str)),
        "citers" => citers(port, rest.first().map(String::as_str), rest.get(1).map(String::as_str)),
        "-h" | "--help" | "help" => {
            println!("{}", usage());
            Ok(())
        }
        "-V" | "--version" => {
            println!("packset {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        other => anyhow::bail!("unknown command: {other}\n\n{}", usage()),
    }
}

fn usage() -> String {
    "packset: start, stop and inspect the pack writer\n\
     \n\
         ensure                 start if down, then print the URL\n\
         start | stop\n\
         status [WORKSPACE]     counts by kind, pin, index\n\
         port | url | which\n\
         pin [NAME]             read, or set, the pinned set\n\
         accessions [WORKSPACE] deed accessions live atoms cite\n\
         citers ACCESSION [WS]  the live atoms citing one accession"
        .to_string()
}

/// The port this seat uses.
fn port() -> u16 {
    env::var("PACKSET_PORT")
        .or_else(|_| env::var("GROK_MEM_PORT"))
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

/// Where the daemon's own output goes.
fn log_path() -> PathBuf {
    if let Some(named) = env::var_os("PACKSET_LOG") {
        return PathBuf::from(named);
    }
    let state = env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("state")))
        .unwrap_or_else(|| PathBuf::from("."));
    state.join("packsetd.log")
}

fn client(port: u16) -> PacksetClient {
    PacksetClient::new(format!("http://127.0.0.1:{port}"))
}

/// Whether the writer on this port is one of ours.
fn is_ours(port: u16) -> bool {
    client(port)
        .health()
        .is_ok_and(|body| OURS.iter().any(|name| body.trim_start().starts_with(name)))
}

/// The workspace a command was given, or the one the environment names.
fn workspace(given: Option<&str>) -> anyhow::Result<String> {
    given
        .map(str::to_string)
        .or_else(|| env::var("PACKSET_WORKSPACE").ok())
        .filter(|w| !w.is_empty())
        .ok_or_else(|| anyhow::anyhow!("name a workspace, or set PACKSET_WORKSPACE"))
}

/// The daemon this seat would run.
///
/// Next to this binary first, because `cargo build` puts the pair in one
/// directory and a checkout should not consult `PATH` to find its own build.
fn resolve_daemon() -> Option<PathBuf> {
    if let Some(named) = env::var_os("PACKSET_BIN") {
        let path = PathBuf::from(named);
        if procfs::executable(&path) {
            return Some(path);
        }
    }
    if let Some(beside) = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("packsetd")))
        .filter(|path| procfs::executable(path))
    {
        return Some(beside);
    }
    env::var_os("PATH")
        .map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .map(|dir| dir.join("packsetd"))
        .find(|path| procfs::executable(path))
}

/// Start the daemon and wait for it to bind.
fn start(port: u16) -> anyhow::Result<()> {
    if procfs::listening(port) {
        if is_ours(port) {
            eprintln!("packset: already listening on 127.0.0.1:{port}");
            return Ok(());
        }
        anyhow::bail!("port {port} is held by something else; set PACKSET_PORT");
    }
    let daemon = resolve_daemon().ok_or_else(|| {
        anyhow::anyhow!("no packsetd binary; cargo build --release -p packset-daemon")
    })?;
    let log = log_path();
    if let Some(dir) = log.parent() {
        fs::create_dir_all(dir)?;
    }
    let out = OpenOptions::new().create(true).append(true).open(&log)?;
    let errs = out.try_clone()?;

    let mut command = Command::new(&daemon);
    command
        .arg("--port")
        .arg(port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(errs));
    // A new session, so closing the terminal that ran `packset ensure` does not
    // take the writer with it.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command.spawn()?;

    let deadline = Instant::now() + STARTUP;
    while Instant::now() < deadline {
        if procfs::listening(port) {
            eprintln!("packset: listening on 127.0.0.1:{port}");
            return Ok(());
        }
        if let Ok(Some(code)) = child.try_wait() {
            anyhow::bail!(
                "packsetd exited {code}; see {}{}",
                log.display(),
                tail(&log)
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    anyhow::bail!("did not come up; see {}{}", log.display(), tail(&log))
}

/// The last few log lines, for an error that would otherwise say only a path.
fn tail(log: &Path) -> String {
    let Ok(text) = fs::read_to_string(log) else {
        return String::new();
    };
    let lines: Vec<&str> = text.lines().rev().take(3).collect();
    if lines.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n");
    for line in lines.into_iter().rev() {
        out.push_str("  ");
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn stop(port: u16) -> anyhow::Result<()> {
    if !is_ours(port) {
        eprintln!("packset: nothing of ours to stop on {port}");
        return Ok(());
    }
    let Some(pid) = procfs::pid_on_port(port) else {
        eprintln!("packset: listening on {port} but the holder is not visible");
        return Ok(());
    };
    procfs::terminate(pid)?;
    eprintln!("packset: stopped {pid}");
    Ok(())
}

fn print_url(port: u16) {
    println!("PACKSET_URL=http://127.0.0.1:{port}");
    println!("INSIDE_MEMORY_URL=http://127.0.0.1:{port}");
}

fn status(port: u16, given: Option<&str>) -> anyhow::Result<()> {
    if !procfs::listening(port) {
        anyhow::bail!("down");
    }
    if !is_ours(port) {
        anyhow::bail!("port {port} is held by another process");
    }
    let client = client(port);
    let health = client.health().unwrap_or_default();
    println!("packset: up on 127.0.0.1:{port} ({})", health.trim());
    let scope = given
        .map(str::to_string)
        .or_else(|| env::var("PACKSET_WORKSPACE").ok())
        .filter(|w| !w.is_empty());
    let detail = client.status(scope.as_deref())?;
    println!("{}", serde_json::to_string_pretty(&detail)?);
    Ok(())
}

fn pin(port: u16, name: Option<&str>) -> anyhow::Result<()> {
    let client = client(port);
    let workspace = workspace(None)?;
    let answer = match name {
        Some(name) => client.set_pin(&workspace, name)?,
        None => client.pin(&workspace)?,
    };
    println!("{}", serde_json::to_string(&answer)?);
    Ok(())
}

/// Every deed accession cited by a live atom, one per line.
///
/// The line-per-accession shape is the point: `deedar evidence -` and `deedar
/// current -` read a list on stdin, so a pack answers the staleness question
/// the same way a tracker does.
/// The live atoms citing one accession, one per line: id, then the claim.
///
/// The other direction of `accessions`, and the pack's half of the backwards
/// walk. A tracker answers which issues cite a product; this answers which
/// remembered claims do.
fn citers(port: u16, accession: Option<&str>, given: Option<&str>) -> anyhow::Result<()> {
    let accession =
        accession.ok_or_else(|| anyhow::anyhow!("name an accession: packset citers ACCESSION"))?;
    let workspace = workspace(given)?;
    for atom in client(port).citers(&workspace, accession)? {
        let id = atom.get("id").and_then(serde_json::Value::as_str).unwrap_or("");
        let text = atom
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        println!("{id}\t{text}");
    }
    Ok(())
}

fn accessions(port: u16, given: Option<&str>) -> anyhow::Result<()> {
    let workspace = workspace(given)?;
    for accession in client(port).accessions(&workspace)? {
        println!("{accession}");
    }
    Ok(())
}
