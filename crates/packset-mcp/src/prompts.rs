//! What the seat knows, asked as a sequence rather than a lookup.
//!
//! A prompt carries what a tool description cannot: an order, and the reason
//! for it. The one sequence this pack is the authority for is crossing from
//! what the seat concluded to what those conclusions stand on, because the
//! accession is the only identifier that leaves this store.
//!
//! Nothing here tells an agent how to read a deed or check a proof. Those are
//! the deed store's formats and it has its own prompts for them.

use rmcp::{
    handler::server::wrapper::Parameters, model::*, prompt, prompt_router, ErrorData as McpError,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::server::PacksetServer;

/// A question to put to the pack.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct QuestionArgs {
    /// What to ask the seat about.
    pub question: String,
    /// The workspace to ask in. Defaults to the one the seat is running.
    pub workspace: Option<String>,
}

fn asked(text: String) -> Vec<PromptMessage> {
    vec![PromptMessage::new_text(Role::User, text)]
}

#[prompt_router(vis = "pub(crate)")]
impl PacksetServer {
    /// Ask what the seat remembers about something, and follow it back to what
    /// the remembered claims stand on.
    #[prompt(name = "what_does_the_seat_know")]
    pub async fn what_does_the_seat_know_prompt(
        &self,
        Parameters(args): Parameters<QuestionArgs>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let question = args.question;
        if question.trim().is_empty() {
            return Err(McpError::invalid_params(
                "a question with no words in it ranks the pack by nothing",
                None,
            ));
        }
        let workspace = args.workspace.filter(|w| !w.trim().is_empty()).map_or_else(
            || "the workspace this seat is running".to_string(),
            |w| format!("workspace {w}"),
        );
        Ok(asked(format!(
            "Ask the pack what this seat knows about: {question}\n\
             \n\
             In {workspace}.\n\
             \n\
             `packset_search` first. Read the result for what it is: ranked recall,\n\
             not a record. An empty list means the pack holds nothing matching. A\n\
             failure means the writer is not running, which is a different thing, and\n\
             `packset_state` is the verb that tells the two apart. Do not report an\n\
             absent writer as an empty pack.\n\
             \n\
             Then cross. A remembered claim is what the seat concluded; the deed is\n\
             what the work produced, and the accession is the only identifier that\n\
             goes between them. `packset_accessions` gives every accession the live\n\
             atoms cite, and `packset_citers` walks it the other way, from a deed back\n\
             to what was concluded from it.\n\
             \n\
             Report what the seat remembers, and separately what any of it stands on.\n\
             A conclusion with no accession behind it is still worth saying and worth\n\
             marking as unbacked."
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every declared prompt renders, from the arguments it says it takes.
    #[tokio::test]
    async fn every_prompt_renders_from_what_it_declares() {
        let declared = PacksetServer::prompt_router().list_all();
        let names: Vec<&str> = declared.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["what_does_the_seat_know"]);
        for prompt in &declared {
            assert!(
                prompt.description.as_ref().is_some_and(|d| !d.is_empty()),
                "{} carries no description",
                prompt.name
            );
            assert!(
                prompt.arguments.as_ref().is_some_and(|a| !a.is_empty()),
                "{} declares no arguments",
                prompt.name
            );
        }

        let server = PacksetServer::at(1, "sample");
        let known = server
            .what_does_the_seat_know_prompt(Parameters(QuestionArgs {
                question: "how the lease works".into(),
                workspace: Some("seat".into()),
            }))
            .await
            .expect("renders");
        let said = format!("{:?}", known[0].content);
        assert!(said.contains("how the lease works"), "{said}");
        assert!(said.contains("workspace seat"), "{said}");
        // The distinction the whole surface exists for survives into the text.
        assert!(said.contains("absent writer"), "{said}");

        let default = server
            .what_does_the_seat_know_prompt(Parameters(QuestionArgs {
                question: "anything".into(),
                workspace: None,
            }))
            .await
            .expect("renders");
        assert!(format!("{:?}", default[0].content).contains("this seat is running"));

        let err = server
            .what_does_the_seat_know_prompt(Parameters(QuestionArgs {
                question: "   ".into(),
                workspace: None,
            }))
            .await
            .expect_err("an empty question rendered");
        assert!(format!("{err:?}").contains("no words in it"), "{err:?}");
    }
}
