//! `packset-mcp`: the pack over the Model Context Protocol.
//!
//! A reader in front of the writer the seat already runs, on the port
//! `PACKSET_PORT` names. Stdio, because a seat runs this beside the agent.

mod args;
mod server;

use rmcp::{transport::stdio, ServiceExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let running = server::PacksetServer::from_env().serve(stdio()).await?;
    running.waiting().await?;
    Ok(())
}
