//! Core types and algorithms for source-grounded information extraction.
//!
//! Behavior is aligned with Google LangExtract's tokenizer, chunking, and
//! alignment pipeline. This is an independent implementation.

pub mod annotation;
pub mod chunking;
pub mod error;
pub mod extract;
pub mod format;
pub mod io;
pub mod model;
pub mod prompt_validation;
pub mod prompting;
pub mod resolver;
pub mod tokenizer;
pub mod types;
pub mod visualization;

pub use error::Error;
pub use extract::{extract, ExtractInput, ExtractOptions, ExtractResult};
pub use io::{load_annotated_documents_jsonl, save_annotated_documents};
pub use tokenizer::{
    find_sentence_range, tokenize, tokens_text, CharInterval, RegexTokenizer, Token, TokenInterval,
    TokenType, TokenizedText, Tokenizer, UnicodeTokenizer,
};
pub use visualization::visualize;
