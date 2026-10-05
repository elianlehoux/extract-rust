//! Shared error types.

use thiserror::Error;

/// Root error for the extraction crate.
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Tokenizer(#[from] crate::tokenizer::TokenizerError),
    #[error(transparent)]
    Chunking(#[from] crate::chunking::ChunkingError),
    #[error(transparent)]
    Format(#[from] crate::format::FormatError),
    #[error(transparent)]
    Resolver(#[from] crate::resolver::ResolverError),
    #[error("{0}")]
    Prompt(String),
    #[error("{0}")]
    InferenceOutput(String),
    #[error("{0}")]
    InvalidDocument(String),
    #[error("{0}")]
    PromptAlignment(String),
    #[error("{0}")]
    Config(String),
    #[error("{0}")]
    Inference(String),
}
