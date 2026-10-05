//! Tokenization used for alignment and sentence-aware chunking.
//!
//! This is not the LLM tokenizer. Offsets are Unicode scalar indices so they
//! match the original string the same way Python `str` indices do.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;
use thiserror::Error;
use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_script::{Script, UnicodeScript};
use unicode_segmentation::UnicodeSegmentation;

/// Default abbreviations that do not end a sentence.
pub const DEFAULT_ABBREVIATIONS: &[&str] = &["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St."];

const CLOSING_PUNCTUATION: &[&str] = &["\"", "'", "\u{201D}", "\u{2019}", "»", ")", "]", "}"];

static TOKEN_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^\W\d_]+|\d+|[^\w\s]|_").expect("token pattern"));
static DIGITS_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d+$").expect("digits pattern"));
static WORD_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:[^\W\d_]+|\d+)$").expect("word pattern"));
static END_OF_SENTENCE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"[.?!。！？\u{0964}]["'\u{201D}\u{2019}»)\]}]*$"#)
        .expect("end of sentence pattern")
});

/// Tokenizer-related errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    #[error("Invalid token interval. start_index={start_index}, end_index={end_index}, total_tokens={total_tokens}.")]
    InvalidTokenInterval {
        start_index: i32,
        end_index: i32,
        total_tokens: usize,
    },
    #[error("start_token_index={start_token_index} out of range. Total tokens: {total_tokens}.")]
    SentenceRange {
        start_token_index: i32,
        total_tokens: usize,
    },
}

/// Character range in the original text (scalar indices, end exclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CharInterval {
    pub start_pos: i32,
    pub end_pos: i32,
}

impl CharInterval {
    pub fn new(start_pos: i32, end_pos: i32) -> Self {
        Self { start_pos, end_pos }
    }
}

/// Token range (end exclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TokenInterval {
    pub start_index: i32,
    pub end_index: i32,
}

impl TokenInterval {
    pub fn new(start_index: i32, end_index: i32) -> Self {
        Self {
            start_index,
            end_index,
        }
    }
}

/// Token class produced by the tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    Word = 0,
    Number = 1,
    Punctuation = 2,
}

/// A token with its character span in the original text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub index: i32,
    pub token_type: TokenType,
    pub char_interval: CharInterval,
    pub first_token_after_newline: bool,
}

impl Token {
    pub fn new(index: i32, token_type: TokenType) -> Self {
        Self {
            index,
            token_type,
            char_interval: CharInterval::new(0, 0),
            first_token_after_newline: false,
        }
    }
}

/// Tokenization result. `text` is the original input, never NFC-normalized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedText {
    pub text: String,
    pub tokens: Vec<Token>,
}

impl TokenizedText {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens: Vec::new(),
        }
    }

    /// Slice of `text` covered by `token`.
    pub fn token_text(&self, token: &Token) -> &str {
        slice_chars(
            &self.text,
            token.char_interval.start_pos,
            token.char_interval.end_pos,
        )
    }
}

/// Splits text into tokens.
pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> TokenizedText;
}

/// Regex tokenizer (default). Faster on Latin text.
#[derive(Debug, Default, Clone, Copy)]
pub struct RegexTokenizer;

impl Tokenizer for RegexTokenizer {
    fn tokenize(&self, text: &str) -> TokenizedText {
        let mut tokenized = TokenizedText::new(text);
        let byte_to_char = byte_to_char_table(text);
        let mut search_at = 0usize;
        let mut previous_end_byte = 0usize;
        let mut token_index: i32 = 0;

        while search_at < text.len() {
            let Some(m) = TOKEN_PATTERN.find_at(text, search_at) else {
                break;
            };
            let mut end_byte = m.end();
            let mut matched = &text[m.start()..end_byte];

            if is_single_symbol(matched) {
                if let Some(ch) = matched.chars().next() {
                    let extra: usize = text[end_byte..]
                        .chars()
                        .take_while(|&c| c == ch)
                        .map(|c| c.len_utf8())
                        .sum();
                    end_byte += extra;
                    matched = &text[m.start()..end_byte];
                }
            }

            let start_pos = byte_to_char[m.start()] as i32;
            let end_pos = byte_to_char[end_byte] as i32;
            let mut token = Token {
                index: token_index,
                token_type: TokenType::Word,
                char_interval: CharInterval::new(start_pos, end_pos),
                first_token_after_newline: false,
            };

            if token_index > 0 {
                let gap = &text[previous_end_byte..m.start()];
                if gap.contains('\n') || gap.contains('\r') {
                    token.first_token_after_newline = true;
                }
            }

            if DIGITS_PATTERN.is_match(matched) {
                token.token_type = TokenType::Number;
            } else if WORD_PATTERN.is_match(matched) {
                token.token_type = TokenType::Word;
            } else {
                token.token_type = TokenType::Punctuation;
            }

            tokenized.tokens.push(token);
            previous_end_byte = end_byte;
            search_at = end_byte;
            token_index += 1;
        }

        tokenized
    }
}

/// Unicode-aware tokenizer. Grapheme clusters, script boundaries, CJK split.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnicodeTokenizer;

impl Tokenizer for UnicodeTokenizer {
    fn tokenize(&self, text: &str) -> TokenizedText {
        let mut tokens: Vec<Token> = Vec::new();
        let mut current_start: Option<usize> = None;
        let mut current_type: Option<TokenType> = None;
        let mut current_script: Option<ScriptKind> = None;
        let mut previous_end: usize = 0;
        let mut char_pos: usize = 0;

        for grapheme in text.graphemes(true) {
            let grapheme_len = grapheme.chars().count();
            let start = char_pos;
            char_pos += grapheme_len;

            if is_whitespace_grapheme(grapheme) {
                if let (Some(ty), Some(cs)) = (current_type, current_start) {
                    emit_token(&mut tokens, text, cs, start, ty, previous_end);
                    previous_end = start;
                    current_type = None;
                    current_script = None;
                    current_start = None;
                }
                continue;
            }

            let g_type = classify_grapheme(grapheme);
            let mut should_merge = false;
            if let Some(cur_ty) = current_type {
                if cur_ty == g_type {
                    match cur_ty {
                        TokenType::Word => {
                            if current_script == Some(ScriptKind::NoGroup) {
                                should_merge = false;
                            } else if let Some(first) = first_scalar(grapheme) {
                                if is_cjk(first) || is_non_spaced(first) {
                                    should_merge = false;
                                } else {
                                    let g_script = script_of(first);
                                    if current_script == Some(g_script)
                                        && g_script != ScriptKind::Unknown
                                    {
                                        should_merge = true;
                                    }
                                }
                            }
                        }
                        TokenType::Number => should_merge = true,
                        TokenType::Punctuation => {
                            if let Some(cs) = current_start {
                                let last = chars_slice(text, cs, start);
                                if last == grapheme
                                    || (last.chars().count() >= grapheme.chars().count()
                                        && last.ends_with(grapheme))
                                {
                                    should_merge = true;
                                }
                            }
                        }
                    }
                }
            }

            if should_merge {
                // extend current token
            } else {
                if let (Some(ty), Some(cs)) = (current_type, current_start) {
                    emit_token(&mut tokens, text, cs, start, ty, previous_end);
                    previous_end = start;
                }
                current_start = Some(start);
                current_type = Some(g_type);
                current_script = if g_type == TokenType::Word {
                    first_scalar(grapheme).map(|c| {
                        if is_cjk(c) || is_non_spaced(c) {
                            ScriptKind::NoGroup
                        } else {
                            script_of(c)
                        }
                    })
                } else {
                    None
                };
            }
        }

        if let (Some(ty), Some(cs)) = (current_type, current_start) {
            emit_token(&mut tokens, text, cs, char_pos, ty, previous_end);
        }

        TokenizedText {
            text: text.to_string(),
            tokens,
        }
    }
}

/// Tokenize with the default [`RegexTokenizer`].
pub fn tokenize(text: &str) -> TokenizedText {
    RegexTokenizer.tokenize(text)
}

/// Reconstruct the original substring spanning `token_interval`.
pub fn tokens_text(
    tokenized_text: &TokenizedText,
    token_interval: TokenInterval,
) -> Result<String, TokenizerError> {
    if token_interval.start_index == token_interval.end_index {
        return Ok(String::new());
    }

    let n = tokenized_text.tokens.len() as i32;
    if token_interval.start_index < 0
        || token_interval.end_index > n
        || token_interval.start_index > token_interval.end_index
    {
        return Err(TokenizerError::InvalidTokenInterval {
            start_index: token_interval.start_index,
            end_index: token_interval.end_index,
            total_tokens: tokenized_text.tokens.len(),
        });
    }

    let start_token = &tokenized_text.tokens[token_interval.start_index as usize];
    let end_token = &tokenized_text.tokens[(token_interval.end_index - 1) as usize];
    Ok(chars_slice(
        &tokenized_text.text,
        start_token.char_interval.start_pos as usize,
        end_token.char_interval.end_pos as usize,
    )
    .to_string())
}

/// Sentence interval starting at `start_token_index`.
pub fn find_sentence_range(
    text: &str,
    tokens: &[Token],
    start_token_index: i32,
    known_abbreviations: Option<&HashSet<String>>,
) -> Result<TokenInterval, TokenizerError> {
    if tokens.is_empty() {
        return Ok(TokenInterval::new(0, 0));
    }

    if start_token_index < 0 || start_token_index as usize >= tokens.len() {
        return Err(TokenizerError::SentenceRange {
            start_token_index,
            total_tokens: tokens.len(),
        });
    }

    let default_abbrevs: HashSet<String> = DEFAULT_ABBREVIATIONS
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let abbrevs = known_abbreviations.unwrap_or(&default_abbrevs);

    let mut i = start_token_index as usize;
    while i < tokens.len() {
        if tokens[i].token_type == TokenType::Punctuation
            && is_end_of_sentence_token(text, tokens, i, abbrevs)
        {
            let mut end_index = i + 1;
            while end_index < tokens.len() {
                let next = token_str(text, &tokens[end_index]);
                if tokens[end_index].token_type == TokenType::Punctuation
                    && CLOSING_PUNCTUATION.contains(&next)
                {
                    end_index += 1;
                } else {
                    break;
                }
            }
            return Ok(TokenInterval::new(
                start_token_index,
                end_index as i32,
            ));
        }
        if is_sentence_break_after_newline(text, tokens, i) {
            return Ok(TokenInterval::new(
                start_token_index,
                (i + 1) as i32,
            ));
        }
        i += 1;
    }

    Ok(TokenInterval::new(
        start_token_index,
        tokens.len() as i32,
    ))
}

fn is_end_of_sentence_token(
    text: &str,
    tokens: &[Token],
    current_idx: usize,
    known_abbreviations: &HashSet<String>,
) -> bool {
    let current = token_str(text, &tokens[current_idx]);
    if END_OF_SENTENCE_PATTERN.is_match(current) {
        if current_idx > 0 {
            let prev = token_str(text, &tokens[current_idx - 1]);
            let combined = format!("{prev}{current}");
            if known_abbreviations.contains(&combined) {
                return false;
            }
        }
        return true;
    }
    false
}

fn is_sentence_break_after_newline(text: &str, tokens: &[Token], current_idx: usize) -> bool {
    if current_idx + 1 >= tokens.len() {
        return false;
    }
    let next = &tokens[current_idx + 1];
    if !next.first_token_after_newline {
        return false;
    }
    let next_text = token_str(text, next);
    if next_text.is_empty() {
        return false;
    }
    !next_text.chars().next().is_some_and(char::is_lowercase)
}

fn emit_token(
    tokens: &mut Vec<Token>,
    text: &str,
    start: usize,
    end: usize,
    token_type: TokenType,
    previous_end: usize,
) {
    let mut first_after_newline = false;
    if start > previous_end {
        let gap = chars_slice(text, previous_end, start);
        if gap.contains('\n') || gap.contains('\r') {
            first_after_newline = true;
        }
    }
    tokens.push(Token {
        index: tokens.len() as i32,
        token_type,
        char_interval: CharInterval::new(start as i32, end as i32),
        first_token_after_newline: first_after_newline,
    });
}

fn classify_grapheme(g: &str) -> TokenType {
    let Some(c) = g.chars().next() else {
        return TokenType::Punctuation;
    };
    match get_general_category(c) {
        GeneralCategory::UppercaseLetter
        | GeneralCategory::LowercaseLetter
        | GeneralCategory::TitlecaseLetter
        | GeneralCategory::ModifierLetter
        | GeneralCategory::OtherLetter => TokenType::Word,
        GeneralCategory::DecimalNumber
        | GeneralCategory::LetterNumber
        | GeneralCategory::OtherNumber => TokenType::Number,
        _ => TokenType::Punctuation,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScriptKind {
    Latin,
    Cyrillic,
    Greek,
    Arabic,
    Hebrew,
    Devanagari,
    NoGroup,
    Unknown,
}

fn script_of(c: char) -> ScriptKind {
    if (c as u32) < 128 {
        return ScriptKind::Latin;
    }
    match c.script() {
        Script::Latin => ScriptKind::Latin,
        Script::Cyrillic => ScriptKind::Cyrillic,
        Script::Greek => ScriptKind::Greek,
        Script::Arabic => ScriptKind::Arabic,
        Script::Hebrew => ScriptKind::Hebrew,
        Script::Devanagari => ScriptKind::Devanagari,
        _ => ScriptKind::Unknown,
    }
}

fn is_cjk(c: char) -> bool {
    matches!(
        c.script(),
        Script::Han | Script::Hiragana | Script::Katakana | Script::Hangul
    )
}

fn is_non_spaced(c: char) -> bool {
    matches!(
        c.script(),
        Script::Thai | Script::Lao | Script::Khmer | Script::Myanmar
    )
}

fn is_whitespace_grapheme(g: &str) -> bool {
    g.chars().all(char::is_whitespace)
}

fn is_single_symbol(matched: &str) -> bool {
    let mut chars = matched.chars();
    let Some(ch) = chars.next() else {
        return false;
    };
    if chars.next().is_some() {
        return false;
    }
    ch == '_' || (!is_word_char(ch) && !ch.is_whitespace())
}

fn is_word_char(c: char) -> bool {
    // Approximate Python regex \w: alphabetic, marks, decimal numbers, connectors.
    if c.is_alphabetic() || c.is_ascii_digit() {
        return true;
    }
    matches!(
        get_general_category(c),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
            | GeneralCategory::DecimalNumber
            | GeneralCategory::ConnectorPunctuation
    ) || c == '\u{200C}'
        || c == '\u{200D}'
}

fn first_scalar(g: &str) -> Option<char> {
    g.chars().next()
}

fn token_str<'a>(text: &'a str, token: &Token) -> &'a str {
    slice_chars(
        text,
        token.char_interval.start_pos,
        token.char_interval.end_pos,
    )
}

fn chars_slice(text: &str, start: usize, end: usize) -> &str {
    let start_byte = char_to_byte(text, start);
    let end_byte = char_to_byte(text, end);
    &text[start_byte..end_byte]
}

fn slice_chars(text: &str, start: i32, end: i32) -> &str {
    let start = start.max(0) as usize;
    let end = end.max(0) as usize;
    chars_slice(text, start, end)
}

fn char_to_byte(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map(|(i, _)| i)
        .unwrap_or(text.len())
}

fn byte_to_char_table(text: &str) -> Vec<usize> {
    let mut table = vec![0usize; text.len() + 1];
    let mut ci = 0usize;
    for (bi, c) in text.char_indices() {
        for k in 0..c.len_utf8() {
            table[bi + k] = ci;
        }
        ci += 1;
    }
    table[text.len()] = ci;
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn types_of(tokens: &[Token]) -> Vec<(i32, TokenType, bool)> {
        tokens
            .iter()
            .map(|t| (t.index, t.token_type, t.first_token_after_newline))
            .collect()
    }

    fn texts_of(tokenized: &TokenizedText) -> Vec<String> {
        tokenized
            .tokens
            .iter()
            .map(|t| tokenized.token_text(t).to_string())
            .collect()
    }

    #[test]
    fn regex_basic_text() {
        let t = tokenize("Hello, world!");
        assert_eq!(
            types_of(&t.tokens),
            vec![
                (0, TokenType::Word, false),
                (1, TokenType::Punctuation, false),
                (2, TokenType::Word, false),
                (3, TokenType::Punctuation, false),
            ]
        );
    }

    #[test]
    fn regex_multiple_spaces_and_numbers() {
        let t = tokenize("Age:   25\nWeight=70kg.");
        assert_eq!(
            types_of(&t.tokens),
            vec![
                (0, TokenType::Word, false),
                (1, TokenType::Punctuation, false),
                (2, TokenType::Number, false),
                (3, TokenType::Word, true),
                (4, TokenType::Punctuation, false),
                (5, TokenType::Number, false),
                (6, TokenType::Word, false),
                (7, TokenType::Punctuation, false),
            ]
        );
    }

    #[test]
    fn regex_multi_line_input() {
        let t = tokenize("Line1\nLine2\nLine3");
        assert_eq!(
            types_of(&t.tokens),
            vec![
                (0, TokenType::Word, false),
                (1, TokenType::Number, false),
                (2, TokenType::Word, true),
                (3, TokenType::Number, false),
                (4, TokenType::Word, true),
                (5, TokenType::Number, false),
            ]
        );
    }

    #[test]
    fn regex_only_symbols() {
        let t = tokenize("!!!@#   $$$%");
        assert_eq!(t.tokens.len(), 5);
        assert!(t.tokens.iter().all(|tok| tok.token_type == TokenType::Punctuation));
        assert_eq!(texts_of(&t), vec!["!!!", "@", "#", "$$$", "%"]);
    }

    #[test]
    fn regex_empty_string() {
        assert!(tokenize("").tokens.is_empty());
    }

    #[test]
    fn regex_non_ascii() {
        let t = tokenize("café");
        assert_eq!(t.tokens.len(), 1);
        assert_eq!(t.tokens[0].token_type, TokenType::Word);
    }

    #[test]
    fn regex_mixed_punctuation() {
        let t = tokenize("?!");
        assert_eq!(t.tokens.len(), 2);
        assert!(t.tokens.iter().all(|tok| tok.token_type == TokenType::Punctuation));
    }

    #[test]
    fn regex_newline_only() {
        assert!(RegexTokenizer.tokenize("\n").tokens.is_empty());
    }

    #[test]
    fn regex_underscore_handling() {
        let t = RegexTokenizer.tokenize("user_id");
        assert_eq!(texts_of(&t), vec!["user", "_", "id"]);
        assert_eq!(t.tokens[0].token_type, TokenType::Word);
        assert_eq!(t.tokens[1].token_type, TokenType::Punctuation);
        assert_eq!(t.tokens[2].token_type, TokenType::Word);
    }

    #[test]
    fn regex_acronym_split() {
        let t = RegexTokenizer.tokenize("A/B");
        assert_eq!(t.tokens.len(), 3);
        assert_eq!(t.tokens[0].token_type, TokenType::Word);
        assert_eq!(t.tokens[1].token_type, TokenType::Punctuation);
        assert_eq!(t.tokens[2].token_type, TokenType::Word);
    }

    #[test]
    fn unicode_japanese() {
        let t = UnicodeTokenizer.tokenize("こんにちは、世界！");
        assert_eq!(t.tokens.len(), 9);
        assert_eq!(t.tokens[5].token_type, TokenType::Punctuation);
        assert_eq!(t.tokens[8].token_type, TokenType::Punctuation);
        assert!(t.tokens[..5].iter().all(|tok| tok.token_type == TokenType::Word));
    }

    #[test]
    fn unicode_english() {
        let t = UnicodeTokenizer.tokenize("Hello, world!");
        assert_eq!(
            types_of(&t.tokens),
            vec![
                (0, TokenType::Word, false),
                (1, TokenType::Punctuation, false),
                (2, TokenType::Word, false),
                (3, TokenType::Punctuation, false),
            ]
        );
    }

    #[test]
    fn unicode_mixed() {
        let t = UnicodeTokenizer.tokenize("Hello 世界 123");
        assert_eq!(
            types_of(&t.tokens),
            vec![
                (0, TokenType::Word, false),
                (1, TokenType::Word, false),
                (2, TokenType::Word, false),
                (3, TokenType::Number, false),
            ]
        );
    }

    #[test]
    fn unicode_mixed_digit_han() {
        let t = UnicodeTokenizer.tokenize("10毫克");
        assert_eq!(texts_of(&t), vec!["10", "毫", "克"]);
        assert_eq!(t.tokens[0].token_type, TokenType::Number);
        assert_eq!(t.tokens[1].token_type, TokenType::Word);
        assert_eq!(t.tokens[2].token_type, TokenType::Word);
    }

    #[test]
    fn unicode_underscore() {
        let t = UnicodeTokenizer.tokenize("hello_world");
        assert_eq!(texts_of(&t), vec!["hello", "_", "world"]);
        let t = UnicodeTokenizer.tokenize("_test_case_");
        assert_eq!(texts_of(&t), vec!["_", "test", "_", "case", "_"]);
    }

    #[test]
    fn unicode_newline_parity() {
        let input = "a\n b";
        assert!(RegexTokenizer.tokenize(input).tokens[1].first_token_after_newline);
        assert!(UnicodeTokenizer.tokenize(input).tokens[1].first_token_after_newline);
    }

    #[test]
    fn unicode_expanded_cjk() {
        let t = UnicodeTokenizer.tokenize("\u{4e00}\u{3400}\u{20000}");
        assert_eq!(t.tokens.len(), 3);
        assert!(t.tokens.iter().all(|tok| tok.token_type == TokenType::Word));
    }

    #[test]
    fn unicode_mixed_script_and_emoji() {
        let t = UnicodeTokenizer.tokenize("Hello👋🏼世界123");
        assert_eq!(texts_of(&t), vec!["Hello", "👋🏼", "世", "界", "123"]);
        assert_eq!(t.tokens[1].token_type, TokenType::Punctuation);
        assert_eq!(t.tokens[4].token_type, TokenType::Number);
    }

    #[test]
    fn unicode_script_boundary() {
        let text = "HelloПривет";
        let t = UnicodeTokenizer.tokenize(text);
        assert_eq!(t.tokens.len(), 2);
        assert_eq!(texts_of(&t), vec!["Hello", "Привет"]);
    }

    #[test]
    fn unicode_thai_no_grouping() {
        let t = UnicodeTokenizer.tokenize("สวัสดี");
        assert_eq!(t.tokens.len(), 4);
    }

    #[test]
    fn unicode_cjk_no_grouping() {
        let t = UnicodeTokenizer.tokenize("你好");
        assert_eq!(t.tokens.len(), 2);
    }

    #[test]
    fn unicode_newline_line_a_b() {
        let t = UnicodeTokenizer.tokenize("LineA\nLineB");
        assert_eq!(t.tokens.len(), 2);
        assert!(!t.tokens[0].first_token_after_newline);
        assert!(t.tokens[1].first_token_after_newline);
    }

    #[test]
    fn unicode_newline_at_start() {
        let t = UnicodeTokenizer.tokenize("\nLineA");
        assert_eq!(t.tokens.len(), 1);
        assert!(t.tokens[0].first_token_after_newline);
    }

    #[test]
    fn unicode_crlf() {
        let t = UnicodeTokenizer.tokenize("LineOne\r\nLineTwo");
        assert_eq!(t.tokens.len(), 2);
        assert!(t.tokens[1].first_token_after_newline);
    }

    #[test]
    fn unicode_unknown_scripts_fragment() {
        let t = UnicodeTokenizer.tokenize("Բարևგამარჯობა");
        assert_eq!(t.tokens.len(), 13);
        let t = UnicodeTokenizer.tokenize("ԲարևԲարև");
        assert_eq!(t.tokens.len(), 8);
    }

    #[test]
    fn unicode_nfd_indices_match_input() {
        let nfd = "e\u{0301}";
        let t = UnicodeTokenizer.tokenize(nfd);
        assert_eq!(t.text, nfd);
        assert_eq!(t.tokens.len(), 1);
        assert_eq!(t.tokens[0].char_interval, CharInterval::new(0, 2));
    }

    #[test]
    fn unicode_identical_punct_groups() {
        let input = "Hello!! World...";
        let t = UnicodeTokenizer.tokenize(input);
        assert_eq!(texts_of(&t), vec!["Hello", "!!", "World", "..."]);
        let mixed = "End.\"";
        let t = UnicodeTokenizer.tokenize(mixed);
        assert_eq!(texts_of(&t), vec!["End", ".", "\""]);
    }

    #[test]
    fn unicode_bengali_devanagari_split() {
        let t = UnicodeTokenizer.tokenize("অअ");
        assert_eq!(t.tokens.len(), 2);
        assert_eq!(t.tokens[0].char_interval, CharInterval::new(0, 1));
        assert_eq!(t.tokens[1].char_interval, CharInterval::new(1, 2));
    }

    #[test]
    fn unicode_identical_unknown_not_merged() {
        let t = UnicodeTokenizer.tokenize("অআ");
        assert_eq!(t.tokens.len(), 2);
    }

    #[test]
    fn tokens_text_valid() {
        let text = "Patient Jane Doe, ID 67890, received 10mg daily.";
        let tok = tokenize(text);
        assert_eq!(
            tokens_text(&tok, TokenInterval::new(1, 3)).unwrap(),
            "Jane Doe"
        );
        assert_eq!(
            tokens_text(&tok, TokenInterval::new(0, 4)).unwrap(),
            "Patient Jane Doe,"
        );
        assert_eq!(
            tokens_text(&tok, TokenInterval::new(5, 6)).unwrap(),
            "67890"
        );
    }

    #[test]
    fn tokens_text_invalid() {
        let tok = tokenize("Patient Jane Doe, ID 67890, received 10mg daily.");
        assert!(tokens_text(&tok, TokenInterval::new(-1, 2)).is_err());
        assert!(tokens_text(&tok, TokenInterval::new(0, 999)).is_err());
        assert!(tokens_text(&tok, TokenInterval::new(5, 4)).is_err());
    }

    #[test]
    fn unicode_tokens_text_errors() {
        let tok = UnicodeTokenizer.tokenize("Hello, world!");
        let err = tokens_text(&tok, TokenInterval::new(-1, 1)).unwrap_err();
        assert!(err.to_string().contains("start_index=-1"));
        let err = tokens_text(&tok, TokenInterval::new(0, 999)).unwrap_err();
        assert!(err.to_string().contains("end_index=999"));
        let err = tokens_text(&tok, TokenInterval::new(2, 1)).unwrap_err();
        assert!(err.to_string().contains("start_index=2"));
        assert!(err.to_string().contains("end_index=1"));
    }

    #[test]
    fn sentence_range_simple() {
        let tok = tokenize("This is one sentence. Then another?");
        let iv = find_sentence_range(&tok.text, &tok.tokens, 0, None).unwrap();
        assert_eq!(iv, TokenInterval::new(0, 5));
    }

    #[test]
    fn sentence_range_abbreviation() {
        let tok = tokenize("Dr. John visited. Then left.");
        let iv = find_sentence_range(&tok.text, &tok.tokens, 0, None).unwrap();
        assert_eq!(iv, TokenInterval::new(0, 5));
    }

    #[test]
    fn sentence_range_newline_capital() {
        let text = "Blood pressure was 160/90 and patient was recommended to\nAtenolol 50 mg daily.";
        let tok = tokenize(text);
        let iv = find_sentence_range(&tok.text, &tok.tokens, 0, None).unwrap();
        assert_eq!(iv, TokenInterval::new(0, 11));
    }

    #[test]
    fn sentence_range_full() {
        let tok = tokenize("Only one sentence here");
        let iv = find_sentence_range(&tok.text, &tok.tokens, 0, None).unwrap();
        assert_eq!(iv.start_index, 0);
        assert_eq!(iv.end_index, tok.tokens.len() as i32);
    }

    #[test]
    fn sentence_range_invalid() {
        let tok = tokenize("Hello world.");
        assert!(find_sentence_range(&tok.text, &tok.tokens, -1, None).is_err());
        assert!(find_sentence_range(&tok.text, &tok.tokens, 999, None).is_err());
        let err = find_sentence_range("Hello world.", &UnicodeTokenizer.tokenize("Hello world.").tokens, 999, None)
            .unwrap_err();
        assert!(err.to_string().contains("Total tokens: 3"));
    }

    #[test]
    fn sentence_range_empty() {
        assert_eq!(
            find_sentence_range("", &[], 0, None).unwrap(),
            TokenInterval::new(0, 0)
        );
    }

    #[test]
    fn sentence_range_quote() {
        let text = "He said \"Hello.\"";
        let tok = UnicodeTokenizer.tokenize(text);
        let iv = find_sentence_range(text, &tok.tokens, 0, None).unwrap();
        assert_eq!(iv.end_index, tok.tokens.len() as i32);
    }

    #[test]
    fn sentence_range_permissive() {
        let text_quote = "\"The time is now.\" Next sentence.";
        let tok = UnicodeTokenizer.tokenize(text_quote);
        let iv = find_sentence_range(text_quote, &tok.tokens, 0, None).unwrap();
        assert_eq!(iv.end_index, 7);

        let text_number = "2025 will be good. Really.";
        let tok = tokenize(text_number);
        let iv = find_sentence_range(text_number, &tok.tokens, 0, None).unwrap();
        assert_eq!(iv.end_index, 5);

        let text_cr = "Line one.\rLine two.";
        let tok = tokenize(text_cr);
        let iv = find_sentence_range(text_cr, &tok.tokens, 0, None).unwrap();
        assert_eq!(iv.end_index, 3);
    }

    #[test]
    fn sentence_range_unicode_terminators() {
        let text_jp = "こんにちは。世界。";
        let tok = UnicodeTokenizer.tokenize(text_jp);
        let iv = find_sentence_range(text_jp, &tok.tokens, 0, None).unwrap();
        assert_eq!(iv.end_index, 6);

        let text_hi = "नमस्ते। दुनिया।";
        let tok = UnicodeTokenizer.tokenize(text_hi);
        let iv = find_sentence_range(text_hi, &tok.tokens, 0, None).unwrap();
        assert_eq!(iv.end_index, 2);
    }

    #[test]
    fn sentence_range_custom_abbreviations() {
        let text_french = "M. Smith est ici.";
        let tok = RegexTokenizer.tokenize(text_french);
        let iv = find_sentence_range(text_french, &tok.tokens, 0, None).unwrap();
        assert_eq!(iv.end_index, 2);

        let mut custom = HashSet::new();
        custom.insert("M.".to_string());
        let iv = find_sentence_range(text_french, &tok.tokens, 0, Some(&custom)).unwrap();
        assert_eq!(iv.end_index, 6);
    }

    #[test]
    fn unicode_invalid_utf8_replacement() {
        let input = "Invalid \u{FFFD} sequence";
        let t = UnicodeTokenizer.tokenize(input);
        assert_eq!(texts_of(&t), vec!["Invalid", "\u{FFFD}", "sequence"]);
        assert_eq!(t.tokens[1].token_type, TokenType::Punctuation);
    }

    #[test]
    fn unicode_long_grapheme() {
        let input = format!("e{}", "\u{0301}".repeat(10));
        let t = UnicodeTokenizer.tokenize(&input);
        assert_eq!(t.tokens.len(), 1);
        assert_eq!(t.tokens[0].token_type, TokenType::Word);
        assert_eq!(t.token_text(&t.tokens[0]), input);
    }

    #[test]
    fn unicode_mixed_valid_invalid() {
        let input = "Valid текст \u{FFFD} 中文";
        let t = UnicodeTokenizer.tokenize(input);
        assert_eq!(texts_of(&t), vec!["Valid", "текст", "\u{FFFD}", "中", "文"]);
    }

    #[test]
    fn unicode_zwj_family() {
        let input = "Family: 👨‍👩‍👧‍👦";
        let t = UnicodeTokenizer.tokenize(input);
        assert_eq!(texts_of(&t), vec!["Family", ":", "👨‍👩‍👧‍👦"]);
        assert_eq!(t.tokens[2].token_type, TokenType::Punctuation);
    }

    #[test]
    fn unicode_isolated_combining_marks() {
        let input = "\u{0301}\u{0302}\u{0303} test";
        let t = UnicodeTokenizer.tokenize(input);
        assert_eq!(texts_of(&t), vec!["\u{0301}\u{0302}\u{0303}", "test"]);
        assert_eq!(t.tokens[0].token_type, TokenType::Punctuation);
    }

    #[test]
    fn unicode_empty_and_whitespace() {
        assert!(UnicodeTokenizer.tokenize("").tokens.is_empty());
        for ws in ["   ", "\t\t", "\n\n", " \t\n\r "] {
            assert!(
                UnicodeTokenizer.tokenize(ws).tokens.is_empty(),
                "whitespace {ws:?} produced tokens"
            );
        }
    }
}
