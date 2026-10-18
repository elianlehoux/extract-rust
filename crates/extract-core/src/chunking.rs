//! Sentence-aware chunking so each piece fits `max_char_buffer`.

use std::fmt;

use regex::Regex;
use thiserror::Error;

use crate::tokenizer::{find_sentence_range, tokens_text, TokenInterval, TokenizedText, Tokenizer};
use crate::types::{CharInterval as DataCharInterval, Document};

/// Errors from chunk construction.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ChunkingError {
    #[error("Start index {start} must be positive.")]
    NegativeStart { start: i32 },
    #[error("Start index {start} must be < end index {end}.")]
    EmptyInterval { start: i32, end: i32 },
    #[error("document_text must be set to access chunk_text.")]
    MissingDocumentText,
    #[error("Token util returns an empty string unexpectedly. Number of tokens is tokenized_text: {n_tokens}, token_interval is {start} to {end}, which should not lead to empty string.")]
    EmptyTokenText {
        n_tokens: usize,
        start: i32,
        end: i32,
    },
    #[error("Sanitized text is empty.")]
    EmptySanitized,
    #[error("Either text or document must be provided.")]
    MissingInput,
    #[error(transparent)]
    Tokenizer(#[from] crate::tokenizer::TokenizerError),
}

/// A chunk of a document, stored as a token interval plus source document.
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub token_interval: TokenInterval,
    pub document: Option<Document>,
}

impl TextChunk {
    pub fn new(token_interval: TokenInterval, document: Option<Document>) -> Self {
        Self {
            token_interval,
            document,
        }
    }

    pub fn document_id(&self) -> Option<&str> {
        self.document.as_ref().map(|d| d.document_id.as_str())
    }

    pub fn document_text(&self) -> Option<&TokenizedText> {
        self.document.as_ref().and_then(|d| d.tokenized_text_ref())
    }

    pub fn additional_context(&self) -> Option<&str> {
        self.document
            .as_ref()
            .and_then(|d| d.additional_context.as_deref())
    }

    pub fn chunk_text(&self) -> Result<String, ChunkingError> {
        let Some(tokenized) = self.document_text() else {
            return Err(ChunkingError::MissingDocumentText);
        };
        get_token_interval_text(tokenized, self.token_interval)
    }

    pub fn sanitized_chunk_text(&self) -> Result<String, ChunkingError> {
        sanitize(&self.chunk_text()?)
    }

    pub fn char_interval(&self) -> Result<DataCharInterval, ChunkingError> {
        let Some(tokenized) = self.document_text() else {
            return Err(ChunkingError::MissingDocumentText);
        };
        get_char_interval(tokenized, self.token_interval)
    }
}

impl fmt::Display for TextChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let interval = format!(
            "start_index: {}, end_index: {}",
            self.token_interval.start_index, self.token_interval.end_index
        );
        let doc_id = match self.document_id() {
            Some(id) => format!("Document ID: {id}"),
            None => "Document ID: None".to_string(),
        };
        let chunk_text = match self.chunk_text() {
            Ok(t) => format!("'{t}'"),
            Err(_) => "<unavailable: document_text not set>".to_string(),
        };
        write!(
            f,
            "TextChunk(\n  interval=[{interval}],\n  {doc_id},\n  Chunk Text: {chunk_text}\n)"
        )
    }
}

impl PartialEq for TextChunk {
    fn eq(&self, other: &Self) -> bool {
        self.token_interval == other.token_interval
            && self.document_id() == other.document_id()
            && self.chunk_text().ok() == other.chunk_text().ok()
    }
}

/// Create a token interval, rejecting empty or negative ranges.
pub fn create_token_interval(
    start_index: i32,
    end_index: i32,
) -> Result<TokenInterval, ChunkingError> {
    if start_index < 0 {
        return Err(ChunkingError::NegativeStart { start: start_index });
    }
    if start_index >= end_index {
        return Err(ChunkingError::EmptyInterval {
            start: start_index,
            end: end_index,
        });
    }
    Ok(TokenInterval::new(start_index, end_index))
}

/// Text covered by `token_interval`.
pub fn get_token_interval_text(
    tokenized_text: &TokenizedText,
    token_interval: TokenInterval,
) -> Result<String, ChunkingError> {
    if token_interval.start_index >= token_interval.end_index {
        return Err(ChunkingError::EmptyInterval {
            start: token_interval.start_index,
            end: token_interval.end_index,
        });
    }
    let s = tokens_text(tokenized_text, token_interval)?;
    if !tokenized_text.text.is_empty() && s.is_empty() {
        return Err(ChunkingError::EmptyTokenText {
            n_tokens: tokenized_text.tokens.len(),
            start: token_interval.start_index,
            end: token_interval.end_index,
        });
    }
    Ok(s)
}

/// Character interval corresponding to a token interval.
pub fn get_char_interval(
    tokenized_text: &TokenizedText,
    token_interval: TokenInterval,
) -> Result<DataCharInterval, ChunkingError> {
    if token_interval.start_index >= token_interval.end_index {
        return Err(ChunkingError::EmptyInterval {
            start: token_interval.start_index,
            end: token_interval.end_index,
        });
    }
    let start_token = &tokenized_text.tokens[token_interval.start_index as usize];
    let final_token = &tokenized_text.tokens[(token_interval.end_index - 1) as usize];
    Ok(DataCharInterval::new(
        start_token.char_interval.start_pos,
        final_token.char_interval.end_pos,
    ))
}

fn sanitize(text: &str) -> Result<String, ChunkingError> {
    static WS: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"\s+").expect("whitespace"));
    let sanitized = WS.replace_all(text.trim(), " ").into_owned();
    if sanitized.is_empty() {
        return Err(ChunkingError::EmptySanitized);
    }
    Ok(sanitized)
}

/// Group chunks into batches of `batch_length`.
pub fn make_batches_of_textchunk(
    chunks: impl IntoIterator<Item = TextChunk>,
    batch_length: usize,
) -> Vec<Vec<TextChunk>> {
    let mut batches = Vec::new();
    let mut current = Vec::new();
    for chunk in chunks {
        current.push(chunk);
        if current.len() == batch_length {
            batches.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// Iterates sentences of a tokenized text.
pub struct SentenceIterator<'a> {
    tokenized_text: &'a TokenizedText,
    curr_token_pos: usize,
}

impl<'a> SentenceIterator<'a> {
    pub fn new(
        tokenized_text: &'a TokenizedText,
        curr_token_pos: usize,
    ) -> Result<Self, ChunkingError> {
        let token_len = tokenized_text.tokens.len();
        if (curr_token_pos as i32) < 0 {
            return Err(ChunkingError::NegativeStart {
                start: curr_token_pos as i32,
            });
        }
        if curr_token_pos > token_len {
            return Err(ChunkingError::EmptyInterval {
                start: curr_token_pos as i32,
                end: token_len as i32,
            });
        }
        Ok(Self {
            tokenized_text,
            curr_token_pos,
        })
    }
}

impl Iterator for SentenceIterator<'_> {
    type Item = TokenInterval;

    fn next(&mut self) -> Option<Self::Item> {
        let token_len = self.tokenized_text.tokens.len();
        if self.curr_token_pos == token_len {
            return None;
        }
        let sentence_range = find_sentence_range(
            &self.tokenized_text.text,
            &self.tokenized_text.tokens,
            self.curr_token_pos as i32,
            None,
        )
        .ok()?;
        let sentence_range =
            create_token_interval(self.curr_token_pos as i32, sentence_range.end_index).ok()?;
        self.curr_token_pos = sentence_range.end_index as usize;
        Some(sentence_range)
    }
}

/// Iterates chunks that fit in `max_char_buffer`.
pub struct ChunkIterator {
    tokenized_text: TokenizedText,
    max_char_buffer: i32,
    curr_token_pos: usize,
    broken_sentence: bool,
    document: Document,
}

impl ChunkIterator {
    pub fn new(
        text: Option<&TokenizedText>,
        raw_text: Option<&str>,
        max_char_buffer: i32,
        tokenizer_impl: &impl Tokenizer,
        document: Option<Document>,
    ) -> Result<Self, ChunkingError> {
        let mut document = match document {
            Some(d) => d,
            None => {
                let t = raw_text
                    .map(str::to_string)
                    .or_else(|| text.map(|t| t.text.clone()))
                    .ok_or(ChunkingError::MissingInput)?;
                Document::new(t)
            }
        };

        let tokenized = if let Some(t) = text {
            if t.tokens.is_empty() {
                let src = if t.text.is_empty() {
                    document.text.clone()
                } else {
                    t.text.clone()
                };
                tokenizer_impl.tokenize(&src)
            } else {
                t.clone()
            }
        } else if let Some(raw) = raw_text {
            tokenizer_impl.tokenize(raw)
        } else {
            tokenizer_impl.tokenize(&document.text)
        };

        document.set_tokenized_text(tokenized.clone());

        Ok(Self {
            tokenized_text: tokenized,
            max_char_buffer,
            curr_token_pos: 0,
            broken_sentence: false,
            document,
        })
    }

    pub fn from_tokenized(
        tokenized_text: TokenizedText,
        max_char_buffer: i32,
        tokenizer_impl: &impl Tokenizer,
        document: Option<Document>,
    ) -> Result<Self, ChunkingError> {
        Self::new(
            Some(&tokenized_text),
            None,
            max_char_buffer,
            tokenizer_impl,
            document,
        )
    }

    pub fn from_str(
        text: &str,
        max_char_buffer: i32,
        tokenizer_impl: &impl Tokenizer,
        document: Option<Document>,
    ) -> Result<Self, ChunkingError> {
        Self::new(None, Some(text), max_char_buffer, tokenizer_impl, document)
    }

    fn tokens_exceed_buffer(&self, token_interval: TokenInterval) -> bool {
        match get_char_interval(&self.tokenized_text, token_interval) {
            Ok(ci) => {
                let start = ci.start_pos.unwrap_or(0);
                let end = ci.end_pos.unwrap_or(0);
                (end - start) > self.max_char_buffer
            }
            Err(_) => false,
        }
    }

    fn next_sentence(&mut self) -> Option<TokenInterval> {
        let mut iter = SentenceIterator {
            tokenized_text: &self.tokenized_text,
            curr_token_pos: self.curr_token_pos,
        };
        let sentence = iter.next()?;
        self.curr_token_pos = iter.curr_token_pos;
        Some(sentence)
    }
}

impl Iterator for ChunkIterator {
    type Item = TextChunk;

    fn next(&mut self) -> Option<Self::Item> {
        let sentence = self.next_sentence()?;
        let mut curr_chunk =
            create_token_interval(sentence.start_index, sentence.start_index + 1).ok()?;

        if self.tokens_exceed_buffer(curr_chunk) {
            self.curr_token_pos = (sentence.start_index + 1) as usize;
            self.broken_sentence = curr_chunk.end_index < sentence.end_index;
            return Some(TextChunk::new(curr_chunk, Some(self.document.clone())));
        }

        let mut start_of_new_line: i32 = -1;
        for token_index in curr_chunk.start_index..sentence.end_index {
            if self.tokenized_text.tokens[token_index as usize].first_token_after_newline {
                start_of_new_line = token_index;
            }
            let test_chunk = create_token_interval(curr_chunk.start_index, token_index + 1).ok()?;
            if self.tokens_exceed_buffer(test_chunk) {
                if start_of_new_line > 0 && start_of_new_line > curr_chunk.start_index {
                    curr_chunk =
                        create_token_interval(curr_chunk.start_index, start_of_new_line).ok()?;
                }
                self.curr_token_pos = curr_chunk.end_index as usize;
                self.broken_sentence = true;
                return Some(TextChunk::new(curr_chunk, Some(self.document.clone())));
            } else {
                curr_chunk = test_chunk;
            }
        }

        if self.broken_sentence {
            self.broken_sentence = false;
        } else {
            while let Some(next_sentence) = self.next_sentence() {
                let test_chunk =
                    create_token_interval(curr_chunk.start_index, next_sentence.end_index).ok()?;
                if self.tokens_exceed_buffer(test_chunk) {
                    self.curr_token_pos = curr_chunk.end_index as usize;
                    return Some(TextChunk::new(curr_chunk, Some(self.document.clone())));
                }
                curr_chunk = test_chunk;
            }
        }

        Some(TextChunk::new(curr_chunk, Some(self.document.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{tokenize, RegexTokenizer, TokenInterval};

    fn interval_text(tokenized: &TokenizedText, iv: TokenInterval) -> String {
        get_token_interval_text(tokenized, iv).unwrap()
    }

    #[test]
    fn sentence_iter_basic() {
        let text = "This is a sentence. This is a longer sentence. Mr. Bond\nasks\nwhy?";
        let tokenized = tokenize(text);
        let mut iter = SentenceIterator::new(&tokenized, 0).unwrap();
        let a = iter.next().unwrap();
        assert_eq!(a, TokenInterval::new(0, 5));
        assert_eq!(interval_text(&tokenized, a), "This is a sentence.");
        let b = iter.next().unwrap();
        assert_eq!(b, TokenInterval::new(5, 11));
        assert_eq!(interval_text(&tokenized, b), "This is a longer sentence.");
        let c = iter.next().unwrap();
        assert_eq!(c, TokenInterval::new(11, 17));
        assert_eq!(interval_text(&tokenized, c), "Mr. Bond\nasks\nwhy?");
        assert!(iter.next().is_none());
    }

    #[test]
    fn sentence_iter_empty() {
        let tokenized = tokenize("");
        let mut iter = SentenceIterator::new(&tokenized, 0).unwrap();
        assert!(iter.next().is_none());
    }

    #[test]
    fn multi_sentence_chunk() {
        let text = "This is a sentence. This is a longer sentence. Mr. Bond\nasks\nwhy?";
        let tokenized = tokenize(text);
        let chunks: Vec<_> =
            ChunkIterator::from_tokenized(tokenized.clone(), 50, &RegexTokenizer, None)
                .unwrap()
                .collect();
        assert_eq!(chunks[0].token_interval, TokenInterval::new(0, 11));
        assert_eq!(
            interval_text(&tokenized, chunks[0].token_interval),
            "This is a sentence. This is a longer sentence."
        );
        assert_eq!(chunks[1].token_interval, TokenInterval::new(11, 17));
        assert_eq!(
            interval_text(&tokenized, chunks[1].token_interval),
            "Mr. Bond\nasks\nwhy?"
        );
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn sentence_with_multiple_newlines() {
        let text = "This is a sentence\n\nThis is a longer sentence\n\nMr\n\nBond\n\nasks why?";
        let tokenized = tokenize(text);
        let iv = TokenInterval::new(0, tokenized.tokens.len() as i32);
        assert_eq!(interval_text(&tokenized, iv), text);
    }

    #[test]
    fn break_sentence() {
        let text = "This is a sentence. This is a longer sentence. Mr. Bond\nasks\nwhy?";
        let tokenized = tokenize(text);
        let chunks: Vec<_> =
            ChunkIterator::from_tokenized(tokenized.clone(), 12, &RegexTokenizer, None)
                .unwrap()
                .collect();
        let texts: Vec<_> = chunks
            .iter()
            .map(|c| interval_text(&tokenized, c.token_interval))
            .collect();
        assert_eq!(texts[0], "This is a");
        assert_eq!(chunks[0].token_interval, TokenInterval::new(0, 3));
        assert_eq!(chunks[1].token_interval, TokenInterval::new(3, 5));
        assert_eq!(texts[1], "sentence.");
        assert_eq!(chunks[2].token_interval, TokenInterval::new(5, 8));
        assert_eq!(texts[2], "This is a");
        assert_eq!(chunks[3].token_interval, TokenInterval::new(8, 9));
        assert_eq!(texts[3], "longer");
        assert_eq!(chunks[4].token_interval, TokenInterval::new(9, 11));
        assert_eq!(texts[4], "sentence.");
        assert_eq!(chunks.len(), 7);
    }

    #[test]
    fn long_token_own_chunk() {
        let text = "This is a sentence. This is a longer sentence. Mr. Bond\nasks\nwhy?";
        let tokenized = tokenize(text);
        let chunks: Vec<_> =
            ChunkIterator::from_tokenized(tokenized.clone(), 7, &RegexTokenizer, None)
                .unwrap()
                .collect();
        assert_eq!(chunks[0].token_interval, TokenInterval::new(0, 2));
        assert_eq!(
            interval_text(&tokenized, chunks[0].token_interval),
            "This is"
        );
        assert_eq!(chunks[1].token_interval, TokenInterval::new(2, 3));
        assert_eq!(interval_text(&tokenized, chunks[1].token_interval), "a");
        assert_eq!(chunks[2].token_interval, TokenInterval::new(3, 4));
        assert_eq!(
            interval_text(&tokenized, chunks[2].token_interval),
            "sentence"
        );
        assert_eq!(chunks[3].token_interval, TokenInterval::new(4, 5));
        assert_eq!(interval_text(&tokenized, chunks[3].token_interval), ".");
        assert_eq!(chunks.len(), 13);
    }

    #[test]
    fn newline_at_chunk_boundary_no_empty() {
        let text = "First sentence.\nSecond sentence that is longer.\nThird sentence.";
        let tokenized = tokenize(text);
        let chunks: Vec<_> = ChunkIterator::from_tokenized(tokenized, 20, &RegexTokenizer, None)
            .unwrap()
            .collect();
        for c in &chunks {
            assert!(c.token_interval.start_index < c.token_interval.end_index);
        }
        let actual: Vec<_> = chunks
            .iter()
            .map(|c| (c.token_interval.start_index, c.token_interval.end_index))
            .collect();
        assert_eq!(actual, vec![(0, 3), (3, 6), (6, 9), (9, 12)]);
    }

    #[test]
    fn chunk_unicode_text() {
        let text = textwrap_dedent(
            "\
    Chief Complaint:
    ‘swelling of tongue and difficulty breathing and swallowing’
    History of Present Illness:
    77 y o woman in NAD with a h/o CAD, DM2, asthma and HTN on altace.",
        );
        let tokenized = tokenize(&text);
        let mut chunks =
            ChunkIterator::from_tokenized(tokenized.clone(), 200, &RegexTokenizer, None).unwrap();
        let chunk = chunks.next().unwrap();
        assert_eq!(
            chunk.token_interval,
            TokenInterval::new(0, tokenized.tokens.len() as i32)
        );
        assert_eq!(interval_text(&tokenized, chunk.token_interval), text);
        assert!(chunks.next().is_none());
    }

    #[test]
    fn newlines_secondary_sentence_break() {
        let text = textwrap_dedent(
            "\
    Medications:
    Theophyline (Uniphyl) 600 mg qhs – bronchodilator by increasing cAMP used
    for treating asthma
    Diltiazem 300 mg qhs – Ca channel blocker used to control hypertension
    Simvistatin (Zocor) 20 mg qhs- HMGCo Reductase inhibitor for
    hypercholesterolemia
    Ramipril (Altace) 10 mg BID – ACEI for hypertension and diabetes for
    renal protective effect",
        );
        let tokenized = tokenize(&text);
        let mut chunks =
            ChunkIterator::from_tokenized(tokenized.clone(), 200, &RegexTokenizer, None).unwrap();
        let first = chunks.next().unwrap();
        let expected_first = textwrap_dedent(
            "\
    Medications:
    Theophyline (Uniphyl) 600 mg qhs – bronchodilator by increasing cAMP used
    for treating asthma
    Diltiazem 300 mg qhs – Ca channel blocker used to control hypertension",
        );
        assert_eq!(
            interval_text(&tokenized, first.token_interval),
            expected_first
        );
        assert!(first.token_interval.end_index > first.token_interval.start_index);
        let second = chunks.next().unwrap();
        let expected_second = textwrap_dedent(
            "\
    Simvistatin (Zocor) 20 mg qhs- HMGCo Reductase inhibitor for
    hypercholesterolemia
    Ramipril (Altace) 10 mg BID – ACEI for hypertension and diabetes for
    renal protective effect",
        );
        assert_eq!(
            interval_text(&tokenized, second.token_interval),
            expected_second
        );
        assert!(chunks.next().is_none());
    }

    #[test]
    fn tokenizer_propagation() {
        let text = "Some text.";
        let mut chunks = ChunkIterator::from_str(text, 100, &RegexTokenizer, None).unwrap();
        let chunk = chunks.next().unwrap();
        assert_eq!(chunk.chunk_text().unwrap(), text);
    }

    #[test]
    fn make_batches_with_data() {
        let text = "Sample text with numerical values such as 120/80 mmHg, 98.6°F, and 50mg.";
        let doc = Document::new(text);
        let tokenized = tokenize(text);
        let chunk_iter =
            ChunkIterator::from_tokenized(tokenized, 10, &RegexTokenizer, Some(doc)).unwrap();
        let batches = make_batches_of_textchunk(chunk_iter, 15);
        assert_eq!(batches.len(), 1);
        let intervals: Vec<_> = batches[0]
            .iter()
            .map(|c| (c.token_interval.start_index, c.token_interval.end_index))
            .collect();
        assert_eq!(
            intervals,
            vec![
                (0, 1),
                (1, 3),
                (3, 4),
                (4, 5),
                (5, 7),
                (7, 10),
                (10, 14),
                (14, 19),
                (19, 22),
            ]
        );
    }

    #[test]
    fn make_batches_empty() {
        let tokenized = tokenize("");
        let chunk_iter =
            ChunkIterator::from_tokenized(tokenized, 10, &RegexTokenizer, None).unwrap();
        let batches = make_batches_of_textchunk(chunk_iter, 15);
        assert!(batches.is_empty());
    }

    #[test]
    fn text_chunk_string_output() {
        let text = "Example input text.";
        let document = Document::new(text).with_document_id("test_doc_123");
        let tokenized = tokenize(text);
        let mut chunks =
            ChunkIterator::from_tokenized(tokenized, 7, &RegexTokenizer, Some(document)).unwrap();
        let chunk = chunks.next().unwrap();
        let expected = "TextChunk(\n  interval=[start_index: 0, end_index: 1],\n  Document ID: test_doc_123,\n  Chunk Text: 'Example'\n)";
        assert_eq!(chunk.to_string(), expected);
    }

    #[test]
    fn additional_context_on_chunks() {
        let ctx = "Some additional context for prompt...";
        let document = Document::new("Sample text.").with_additional_context(ctx);
        let tokenized = tokenize("Sample text.");
        let mut chunks =
            ChunkIterator::from_tokenized(tokenized, 100, &RegexTokenizer, Some(document)).unwrap();
        assert_eq!(chunks.next().unwrap().additional_context(), Some(ctx));

        let document = Document::new("Sample text.");
        let tokenized = tokenize("Sample text.");
        let mut chunks =
            ChunkIterator::from_tokenized(tokenized, 100, &RegexTokenizer, Some(document)).unwrap();
        assert_eq!(chunks.next().unwrap().additional_context(), None);

        let text = "Sentence one. Sentence two. Sentence three.";
        let document = Document::new(text).with_additional_context(ctx);
        let tokenized = tokenize(text);
        let chunks: Vec<_> =
            ChunkIterator::from_tokenized(tokenized, 15, &RegexTokenizer, Some(document))
                .unwrap()
                .collect();
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.additional_context() == Some(ctx)));
    }

    #[test]
    fn text_chunk_properties_with_document() {
        let mut document = Document::new("Sample text.")
            .with_document_id("doc123")
            .with_additional_context("Additional info");
        document.set_tokenized_text(tokenize("Sample text."));
        let chunk = TextChunk::new(TokenInterval::new(0, 1), Some(document));
        assert_eq!(chunk.document_id(), Some("doc123"));
        assert_eq!(chunk.document_text().unwrap().text, "Sample text.");
        assert_eq!(chunk.additional_context(), Some("Additional info"));
    }

    #[test]
    fn text_chunk_properties_no_document() {
        let chunk = TextChunk::new(TokenInterval::new(0, 1), None);
        assert_eq!(chunk.document_id(), None);
        assert!(chunk.document_text().is_none());
        assert_eq!(chunk.additional_context(), None);
    }

    fn textwrap_dedent(s: &str) -> String {
        // Match Python textwrap.dedent: strip common leading whitespace per line.
        let lines: Vec<&str> = s.split('\n').collect();
        let indent = lines
            .iter()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.chars().take_while(|c| c.is_whitespace()).count())
            .min()
            .unwrap_or(0);
        lines
            .iter()
            .map(|l| {
                let skip = indent.min(l.chars().count());
                l.chars().skip(skip).collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
