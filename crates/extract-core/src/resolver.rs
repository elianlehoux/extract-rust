//! Parse LLM output and align extractions back to source offsets.

use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::{Map, Value};
use thiserror::Error;

use crate::format::{FormatError, FormatHandler};
use crate::tokenizer::{RegexTokenizer, TokenInterval, TokenizedText, Tokenizer};
use crate::types::{AlignmentStatus, AttributeValue, CharInterval, Extraction, FormatType};

pub const DEFAULT_INDEX_SUFFIX: &str = "_index";
pub const DEFAULT_DELIM: &str = "\u{241F}";
pub const FUZZY_ALIGNMENT_MIN_THRESHOLD: f64 = 0.75;
pub const FUZZY_ALIGNMENT_MIN_DENSITY: f64 = 1.0 / 3.0;

const FUZZY_LCS: &str = "lcs";
const FUZZY_LEGACY: &str = "legacy";
const EXACT_DP: &str = "dp";
const EXACT_DIFFLIB: &str = "difflib";

/// Knobs for exact + fuzzy alignment.
#[derive(Clone)]
pub struct AlignOptions<'a> {
    pub token_offset: i32,
    pub char_offset: i32,
    pub delim: &'a str,
    pub enable_fuzzy_alignment: bool,
    pub fuzzy_alignment_threshold: f64,
    pub accept_match_lesser: bool,
    pub tokenizer: Option<&'a dyn Tokenizer>,
    pub fuzzy_alignment_algorithm: &'a str,
    pub fuzzy_alignment_min_density: f64,
    pub exact_alignment_algorithm: &'a str,
}

impl Default for AlignOptions<'_> {
    fn default() -> Self {
        Self {
            token_offset: 0,
            char_offset: 0,
            delim: DEFAULT_DELIM,
            enable_fuzzy_alignment: true,
            fuzzy_alignment_threshold: FUZZY_ALIGNMENT_MIN_THRESHOLD,
            accept_match_lesser: true,
            tokenizer: None,
            fuzzy_alignment_algorithm: FUZZY_LCS,
            fuzzy_alignment_min_density: FUZZY_ALIGNMENT_MIN_DENSITY,
            exact_alignment_algorithm: EXACT_DP,
        }
    }
}

/// Parse/alignment errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ResolverError {
    #[error("{0}")]
    Parse(String),
    #[error("{0}")]
    Schema(String),
    #[error("{0}")]
    Align(String),
}

impl From<FormatError> for ResolverError {
    fn from(e: FormatError) -> Self {
        Self::Parse(e.to_string())
    }
}

/// Tightest source span for an LCS match count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LcsSpan {
    pub matches: i32,
    pub start: i32,
    pub end: i32,
}

impl LcsSpan {
    pub fn span_len(self) -> i32 {
        if self.start < 0 {
            0
        } else {
            self.end - self.start + 1
        }
    }
}

const NO_MATCH: LcsSpan = LcsSpan {
    matches: 0,
    start: -1,
    end: -1,
};

/// Parses model output into ordered [`Extraction`]s.
#[derive(Debug, Clone)]
pub struct Resolver {
    pub format_handler: FormatHandler,
    pub extraction_index_suffix: Option<String>,
    pub strict: bool,
}

impl Resolver {
    pub fn new(format_handler: FormatHandler, extraction_index_suffix: Option<String>) -> Self {
        Self {
            format_handler,
            extraction_index_suffix,
            strict: false,
        }
    }

    pub fn from_legacy(
        fence_output: bool,
        format_type: FormatType,
        extraction_index_suffix: Option<String>,
    ) -> Self {
        Self::new(
            FormatHandler::from_kwargs(Some(fence_output), Some(format_type), None, None, None),
            extraction_index_suffix,
        )
    }

    pub fn resolve(
        &self,
        input_text: &str,
        suppress_parse_errors: bool,
    ) -> Result<Vec<Extraction>, ResolverError> {
        let extraction_data = match self.format_handler.parse_output(input_text, self.strict) {
            Ok(data) => data,
            Err(_e) if suppress_parse_errors => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        match self.extract_ordered_extractions(&extraction_data) {
            Ok(v) => Ok(v),
            Err(_e) if suppress_parse_errors => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    pub fn extract_ordered_extractions(
        &self,
        extraction_data: &[Map<String, Value>],
    ) -> Result<Vec<Extraction>, ResolverError> {
        let mut processed = Vec::new();
        let mut extraction_index: i32 = 0;
        let index_suffix = self.extraction_index_suffix.as_deref();
        let attributes_suffix = self.format_handler.attribute_suffix.as_str();

        for (group_index, group) in extraction_data.iter().enumerate() {
            for (extraction_class, extraction_value) in group {
                if let Some(suffix) = index_suffix {
                    if extraction_class.ends_with(suffix) {
                        if !is_json_int(extraction_value) {
                            return Err(ResolverError::Schema(
                                "Index must be an integer.".to_string(),
                            ));
                        }
                        continue;
                    }
                }
                if extraction_class.ends_with(attributes_suffix) {
                    if !(extraction_value.is_object() || extraction_value.is_null()) {
                        return Err(ResolverError::Schema(
                            "Extraction value must be a dict or None for attributes.".to_string(),
                        ));
                    }
                    continue;
                }

                let text = value_to_extraction_text(extraction_value)?;
                let resolved_index = if let Some(suffix) = index_suffix {
                    let index_key = format!("{extraction_class}{suffix}");
                    match group.get(&index_key) {
                        Some(v) => json_int(v).ok_or_else(|| {
                            ResolverError::Schema("Index must be an integer.".to_string())
                        })?,
                        None => continue,
                    }
                } else {
                    extraction_index += 1;
                    extraction_index
                };

                let attributes = {
                    let attributes_key = format!("{extraction_class}{attributes_suffix}");
                    group.get(&attributes_key).and_then(json_attributes)
                };

                let mut ext = Extraction::new(extraction_class, text)
                    .with_index(resolved_index)
                    .with_group(group_index as i32);
                ext.attributes = attributes;
                processed.push(ext);
            }
        }
        processed.sort_by_key(|e| e.extraction_index.unwrap_or(0));
        Ok(processed)
    }

    pub fn align(
        &self,
        extractions: Vec<Extraction>,
        source_text: &str,
        options: AlignOptions<'_>,
    ) -> Result<Vec<Extraction>, ResolverError> {
        if extractions.is_empty() {
            return Ok(Vec::new());
        }
        let groups = WordAligner.align_groups(vec![extractions], source_text, options)?;
        Ok(groups.into_iter().flatten().collect())
    }
}

/// Aligns extraction strings to token/char intervals in the source text.
#[derive(Debug, Default, Clone, Copy)]
pub struct WordAligner;

impl WordAligner {
    pub fn align_extractions(
        self,
        extraction_groups: Vec<Vec<Extraction>>,
        source_text: &str,
        options: AlignOptions<'_>,
    ) -> Result<Vec<Vec<Extraction>>, ResolverError> {
        self.align_groups(extraction_groups, source_text, options)
    }

    /// Align grouped extractions.
    pub fn align_groups(
        self,
        mut extraction_groups: Vec<Vec<Extraction>>,
        source_text: &str,
        options: AlignOptions<'_>,
    ) -> Result<Vec<Vec<Extraction>>, ResolverError> {
        let AlignOptions {
            token_offset,
            char_offset,
            delim,
            enable_fuzzy_alignment,
            fuzzy_alignment_threshold,
            accept_match_lesser,
            tokenizer: tokenizer_impl,
            fuzzy_alignment_algorithm,
            fuzzy_alignment_min_density,
            exact_alignment_algorithm,
        } = options;
        if exact_alignment_algorithm != EXACT_DP && exact_alignment_algorithm != EXACT_DIFFLIB {
            return Err(ResolverError::Align(format!(
                "Invalid exact_alignment_algorithm '{exact_alignment_algorithm}'; expected one of [\"difflib\", \"dp\"]."
            )));
        }
        if enable_fuzzy_alignment {
            if fuzzy_alignment_algorithm != FUZZY_LCS && fuzzy_alignment_algorithm != FUZZY_LEGACY {
                return Err(ResolverError::Align(format!(
                    "Invalid fuzzy_alignment_algorithm '{fuzzy_alignment_algorithm}'."
                )));
            }
            if !(0.0..=1.0).contains(&fuzzy_alignment_threshold) {
                return Err(ResolverError::Align(
                    "fuzzy_alignment_threshold must be in [0, 1]".to_string(),
                ));
            }
            if !(0.0..=1.0).contains(&fuzzy_alignment_min_density) {
                return Err(ResolverError::Align(
                    "fuzzy_alignment_min_density must be in [0, 1]".to_string(),
                ));
            }
        }
        if extraction_groups.is_empty() {
            return Ok(Vec::new());
        }

        let tok: &dyn Tokenizer = tokenizer_impl.unwrap_or(&RegexTokenizer);
        let source_tokens = tokenize_with_lowercase(source_text, tok);
        let delim_tokens = tokenize_with_lowercase(delim, tok);
        if delim_tokens.len() != 1 {
            return Err(ResolverError::Align(format!(
                "Delimiter {delim:?} must be a single token."
            )));
        }
        let delim_len = delim_tokens.len();

        for group in &extraction_groups {
            for extraction in group {
                if extraction.extraction_text.contains(delim) {
                    return Err(ResolverError::Align(format!(
                        "Delimiter {delim:?} appears inside extraction text {:?}.",
                        extraction.extraction_text
                    )));
                }
            }
        }

        let joined = extraction_groups
            .iter()
            .flatten()
            .map(|e| e.extraction_text.as_str())
            .collect::<Vec<_>>()
            .join(&format!(" {delim} "));
        let extraction_tokens = tokenize_with_lowercase(&joined, tok);
        if source_tokens.is_empty() || extraction_tokens.is_empty() {
            return Err(ResolverError::Align(
                "Source tokens and extraction tokens cannot be empty.".to_string(),
            ));
        }

        // Map start-token index in the concatenated extraction stream -> (group, idx).
        let mut index_to_ref: HashMap<usize, (usize, usize)> = HashMap::new();
        let mut extraction_index = 0usize;
        for (g, group) in extraction_groups.iter().enumerate() {
            for (i, extraction) in group.iter().enumerate() {
                index_to_ref.insert(extraction_index, (g, i));
                let n = tokenize_with_lowercase(&extraction.extraction_text, tok).len();
                extraction_index += n + delim_len;
            }
        }

        let tokenized_text = tok.tokenize(source_text);

        let mut dp_matched: HashSet<(usize, usize)> = HashSet::new();
        if exact_alignment_algorithm == EXACT_DP {
            let ordered: Vec<(usize, usize)> = extraction_groups
                .iter()
                .enumerate()
                .flat_map(|(g, group)| (0..group.len()).map(move |i| (g, i)))
                .collect();
            let matched = apply_monotonic_exact_matches(
                &mut extraction_groups,
                &ordered,
                &source_tokens,
                &tokenized_text,
                token_offset,
                char_offset,
                tok,
            );
            dp_matched.extend(matched);
        }

        let blocks = matching_blocks(&source_tokens, &extraction_tokens);
        for &(i, j, n) in &blocks {
            if n == 0 {
                continue;
            }
            let Some(&(g, idx)) = index_to_ref.get(&j) else {
                continue;
            };
            if dp_matched.contains(&(g, idx)) {
                continue;
            }
            let extraction = &mut extraction_groups[g][idx];
            extraction.set_token_interval(Some(TokenInterval::new(
                i as i32 + token_offset,
                (i + n) as i32 + token_offset,
            )));
            let start_token = &tokenized_text.tokens[i];
            let end_token = &tokenized_text.tokens[i + n - 1];
            extraction.char_interval = Some(CharInterval::new(
                char_offset + start_token.char_interval.start_pos,
                char_offset + end_token.char_interval.end_pos,
            ));
            let extraction_text_len =
                tokenize_with_lowercase(&extraction.extraction_text, tok).len();
            if extraction_text_len < n {
                return Err(ResolverError::Align(
                    "Delimiter prevents blocks greater than extraction length".to_string(),
                ));
            }
            if extraction_text_len == n {
                extraction.alignment_status = Some(AlignmentStatus::MatchExact);
            } else if accept_match_lesser {
                extraction.alignment_status = Some(AlignmentStatus::MatchLesser);
            } else {
                extraction.set_token_interval(None);
                extraction.char_interval = None;
                extraction.alignment_status = None;
            }
        }

        if enable_fuzzy_alignment {
            let src_norm: Option<Vec<String>> = if fuzzy_alignment_algorithm == FUZZY_LCS {
                Some(source_tokens.iter().map(|t| normalize_token(t)).collect())
            } else {
                None
            };
            let unaligned: Vec<(usize, usize)> = extraction_groups
                .iter()
                .enumerate()
                .flat_map(|(g, group)| {
                    group
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| e.alignment_status.is_none())
                        .map(move |(i, _)| (g, i))
                })
                .collect();
            for (g, idx) in unaligned {
                let aligned = if fuzzy_alignment_algorithm == FUZZY_LCS {
                    lcs_fuzzy_align_extraction(
                        &extraction_groups[g][idx],
                        src_norm.as_ref().unwrap(),
                        &tokenized_text,
                        token_offset,
                        char_offset,
                        fuzzy_alignment_threshold,
                        fuzzy_alignment_min_density,
                        tok,
                    )
                } else {
                    fuzzy_align_extraction(
                        &extraction_groups[g][idx],
                        &source_tokens,
                        &tokenized_text,
                        token_offset,
                        char_offset,
                        fuzzy_alignment_threshold,
                        tok,
                    )
                };
                if let Some(updated) = aligned {
                    extraction_groups[g][idx] = updated;
                }
            }
        }

        Ok(extraction_groups)
    }

    pub fn lcs_fuzzy_align(
        &self,
        extraction: &Extraction,
        source_tokens_norm: &[String],
        tokenized_text: &TokenizedText,
        token_offset: i32,
        char_offset: i32,
        tokenizer_impl: &dyn Tokenizer,
    ) -> Option<Extraction> {
        lcs_fuzzy_align_extraction(
            extraction,
            source_tokens_norm,
            tokenized_text,
            token_offset,
            char_offset,
            FUZZY_ALIGNMENT_MIN_THRESHOLD,
            FUZZY_ALIGNMENT_MIN_DENSITY,
            tokenizer_impl,
        )
    }

    pub fn legacy_fuzzy_align(
        &self,
        extraction: &Extraction,
        source_tokens: &[String],
        tokenized_text: &TokenizedText,
        token_offset: i32,
        char_offset: i32,
        tokenizer_impl: &dyn Tokenizer,
    ) -> Option<Extraction> {
        fuzzy_align_extraction(
            extraction,
            source_tokens,
            tokenized_text,
            token_offset,
            char_offset,
            FUZZY_ALIGNMENT_MIN_THRESHOLD,
            tokenizer_impl,
        )
    }
}

impl WordAligner {
    pub fn align(
        self,
        extraction_groups: Vec<Vec<Extraction>>,
        source_text: &str,
        token_offset: i32,
        char_offset: i32,
    ) -> Result<Vec<Vec<Extraction>>, ResolverError> {
        self.align_groups(
            extraction_groups,
            source_text,
            AlignOptions {
                token_offset,
                char_offset,
                ..AlignOptions::default()
            },
        )
    }
}

/// Latest-start LCS spans keyed by match count.
pub fn best_lcs_spans(source: &[String], extraction: &[String]) -> HashMap<i32, LcsSpan> {
    let n = source.len();
    let m = extraction.len();
    if n == 0 || m == 0 {
        return HashMap::new();
    }
    let dim = m + 1;
    let mut prev_row = vec![-1i32; dim * dim];
    let mut curr_row = vec![-1i32; dim * dim];
    let idx = |j: usize, k: usize| j * dim + k;
    for j in 0..=m {
        prev_row[idx(j, 0)] = 0;
    }
    let mut best_per_k: HashMap<i32, LcsSpan> = HashMap::new();

    for i in 1..=n {
        let src_tok = &source[i - 1];
        curr_row[idx(0, 0)] = i as i32;
        for k in 1..=m {
            curr_row[idx(0, k)] = -1;
        }
        for j in 1..=m {
            curr_row[idx(j, 0)] = i as i32;
            let matches_here = src_tok == &extraction[j - 1];
            for k in 1..=m {
                let skip_source = prev_row[idx(j, k)];
                let skip_extraction = curr_row[idx(j - 1, k)];
                let mut best = skip_source.max(skip_extraction);
                if matches_here {
                    let match_cand = if k == 1 {
                        (i as i32) - 1
                    } else {
                        prev_row[idx(j - 1, k - 1)]
                    };
                    best = best.max(match_cand);
                }
                curr_row[idx(j, k)] = best;
            }
        }
        let end = (i as i32) - 1;
        for k in 1..=m {
            let s = curr_row[idx(m, k)];
            if s < 0 {
                continue;
            }
            let k_i = k as i32;
            let candidate = LcsSpan {
                matches: k_i,
                start: s,
                end,
            };
            match best_per_k.get(&k_i) {
                None => {
                    best_per_k.insert(k_i, candidate);
                }
                Some(existing) => {
                    let new_len = end - s + 1;
                    let cur_len = existing.span_len();
                    if new_len < cur_len || (new_len == cur_len && s < existing.start) {
                        best_per_k.insert(k_i, candidate);
                    }
                }
            }
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }
    best_per_k
}

pub fn best_lcs_span(source: &[String], extraction: &[String]) -> LcsSpan {
    let spans = best_lcs_spans(source, extraction);
    spans
        .into_iter()
        .max_by_key(|(k, _)| *k)
        .map(|(_, s)| s)
        .unwrap_or(NO_MATCH)
}

pub fn accept_lcs_match(
    span: LcsSpan,
    extraction_len: i32,
    threshold: f64,
    min_density: f64,
) -> bool {
    if span.matches == 0 || extraction_len == 0 {
        return false;
    }
    let needed = (extraction_len as f64 * threshold).ceil() as i32;
    if span.span_len() <= 0 {
        return false;
    }
    let density = span.matches as f64 / span.span_len() as f64;
    span.matches >= needed && density >= min_density
}

pub fn normalize_token(token: &str) -> String {
    let token = token.to_lowercase();
    let chars: Vec<char> = token.chars().collect();
    if chars.len() > 3 && token.ends_with('s') && !token.ends_with("ss") {
        chars[..chars.len() - 1].iter().collect()
    } else {
        token
    }
}

fn tokenize_with_lowercase(text: &str, tokenizer: &dyn Tokenizer) -> Vec<String> {
    let tokenized = tokenizer.tokenize(text);
    tokenized
        .tokens
        .iter()
        .map(|t| tokenized.token_text(t).to_lowercase())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn lcs_fuzzy_align_extraction(
    extraction: &Extraction,
    source_tokens_norm: &[String],
    tokenized_text: &TokenizedText,
    token_offset: i32,
    char_offset: i32,
    fuzzy_alignment_threshold: f64,
    fuzzy_alignment_min_density: f64,
    tokenizer_impl: &dyn Tokenizer,
) -> Option<Extraction> {
    let extraction_tokens = tokenize_with_lowercase(&extraction.extraction_text, tokenizer_impl);
    if extraction_tokens.is_empty() {
        return None;
    }
    let extraction_tokens_norm: Vec<String> = extraction_tokens
        .iter()
        .map(|t| normalize_token(t))
        .collect();
    let spans = best_lcs_spans(source_tokens_norm, &extraction_tokens_norm);
    let mut ks: Vec<i32> = spans.keys().copied().collect();
    ks.sort_by(|a, b| b.cmp(a));
    let mut accepted = None;
    for k in ks {
        let candidate = spans[&k];
        if accept_lcs_match(
            candidate,
            extraction_tokens_norm.len() as i32,
            fuzzy_alignment_threshold,
            fuzzy_alignment_min_density,
        ) {
            accepted = Some(candidate);
            break;
        }
    }
    let accepted = accepted?;
    let mut out = extraction.clone();
    out.set_token_interval(Some(TokenInterval::new(
        accepted.start + token_offset,
        accepted.end + 1 + token_offset,
    )));
    let start_token = &tokenized_text.tokens[accepted.start as usize];
    let end_token = &tokenized_text.tokens[accepted.end as usize];
    out.char_interval = Some(CharInterval::new(
        char_offset + start_token.char_interval.start_pos,
        char_offset + end_token.char_interval.end_pos,
    ));
    out.alignment_status = Some(AlignmentStatus::MatchFuzzy);
    Some(out)
}

#[allow(clippy::too_many_arguments)]
fn fuzzy_align_extraction(
    extraction: &Extraction,
    source_tokens: &[String],
    tokenized_text: &TokenizedText,
    token_offset: i32,
    char_offset: i32,
    fuzzy_alignment_threshold: f64,
    tokenizer_impl: &dyn Tokenizer,
) -> Option<Extraction> {
    let extraction_tokens = tokenize_with_lowercase(&extraction.extraction_text, tokenizer_impl);
    let extraction_tokens_norm: Vec<String> = extraction_tokens
        .iter()
        .map(|t| normalize_token(t))
        .collect();
    if extraction_tokens.is_empty() {
        return None;
    }
    let len_e = extraction_tokens.len();
    let max_window = source_tokens.len();
    let extraction_counts = count_tokens(&extraction_tokens_norm);
    let min_overlap = (len_e as f64 * fuzzy_alignment_threshold) as usize;

    let mut best_ratio = 0.0f64;
    let mut best_span: Option<(usize, usize)> = None;

    for window_size in len_e..=max_window {
        if window_size > source_tokens.len() {
            break;
        }
        let mut window: VecDeque<String> = source_tokens[..window_size].iter().cloned().collect();
        let mut window_counts = count_tokens(
            &window
                .iter()
                .map(|t| normalize_token(t))
                .collect::<Vec<_>>(),
        );
        for start_idx in 0..=source_tokens.len() - window_size {
            if intersection_total(&extraction_counts, &window_counts) >= min_overlap {
                let window_tokens_norm: Vec<String> =
                    window.iter().map(|t| normalize_token(t)).collect();
                let blocks = matching_blocks(&window_tokens_norm, &extraction_tokens_norm);
                let matches: usize = blocks.iter().map(|b| b.2).sum();
                let ratio = if len_e > 0 {
                    matches as f64 / len_e as f64
                } else {
                    0.0
                };
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best_span = Some((start_idx, window_size));
                }
            }
            if start_idx + window_size < source_tokens.len() {
                if let Some(old) = window.pop_front() {
                    let old_n = normalize_token(&old);
                    if let Some(c) = window_counts.get_mut(&old_n) {
                        *c -= 1;
                        if *c == 0 {
                            window_counts.remove(&old_n);
                        }
                    }
                }
                let new_token = source_tokens[start_idx + window_size].clone();
                let new_n = normalize_token(&new_token);
                *window_counts.entry(new_n).or_insert(0) += 1;
                window.push_back(new_token);
            }
        }
    }

    if let Some((start_idx, window_size)) = best_span {
        if best_ratio >= fuzzy_alignment_threshold {
            let mut out = extraction.clone();
            out.set_token_interval(Some(TokenInterval::new(
                start_idx as i32 + token_offset,
                (start_idx + window_size) as i32 + token_offset,
            )));
            let start_token = tokenized_text.tokens.get(start_idx)?;
            let end_token = tokenized_text.tokens.get(start_idx + window_size - 1)?;
            out.char_interval = Some(CharInterval::new(
                char_offset + start_token.char_interval.start_pos,
                char_offset + end_token.char_interval.end_pos,
            ));
            out.alignment_status = Some(AlignmentStatus::MatchFuzzy);
            return Some(out);
        }
    }
    None
}

fn count_tokens(tokens: &[String]) -> HashMap<String, usize> {
    let mut c = HashMap::new();
    for t in tokens {
        *c.entry(t.clone()).or_insert(0) += 1;
    }
    c
}

fn intersection_total(a: &HashMap<String, usize>, b: &HashMap<String, usize>) -> usize {
    a.iter().map(|(k, va)| va.min(b.get(k).unwrap_or(&0))).sum()
}

fn find_token_occurrences(source: &[String], extraction: &[String]) -> Vec<usize> {
    let n = source.len();
    let m = extraction.len();
    if m == 0 || n < m {
        return Vec::new();
    }
    let first = &extraction[0];
    (0..=n - m)
        .filter(|&i| &source[i] == first && source[i..i + m] == extraction[..])
        .collect()
}

struct ChainNode {
    extraction_index: usize,
    start: usize,
    parent: Option<usize>,
}

struct FrontierEntry {
    end: i32,
    weight: i32,
    node: usize,
}

fn select_monotonic_matches(
    occurrence_lists: &[Vec<usize>],
    extraction_lengths: &[usize],
) -> HashMap<usize, usize> {
    let mut frontier: Vec<FrontierEntry> = Vec::new();
    let mut nodes: Vec<ChainNode> = Vec::new();

    let best_ending_at_or_before = |frontier: &[FrontierEntry], position: i32| -> Option<usize> {
        match frontier.binary_search_by(|e| e.end.cmp(&position)) {
            Ok(i) => Some(i),
            Err(i) => {
                if i == 0 {
                    None
                } else {
                    Some(i - 1)
                }
            }
        }
    };

    for (extraction_index, (occurrences, &length)) in occurrence_lists
        .iter()
        .zip(extraction_lengths.iter())
        .enumerate()
    {
        if occurrences.is_empty() || length == 0 {
            continue;
        }
        let mut candidates = Vec::new();
        for &start in occurrences {
            let pred = best_ending_at_or_before(&frontier, start as i32);
            let weight = length as i32 + pred.map(|i| frontier[i].weight).unwrap_or(0);
            let parent = pred.map(|i| frontier[i].node);
            nodes.push(ChainNode {
                extraction_index,
                start,
                parent,
            });
            candidates.push(FrontierEntry {
                end: (start + length) as i32,
                weight,
                node: nodes.len() - 1,
            });
        }
        for candidate in candidates {
            let covering = best_ending_at_or_before(&frontier, candidate.end);
            if covering.is_some_and(|i| frontier[i].weight >= candidate.weight) {
                continue;
            }
            let low = match frontier.binary_search_by(|e| e.end.cmp(&candidate.end)) {
                Ok(i) | Err(i) => i,
            };
            let mut dominated_end = low;
            while dominated_end < frontier.len()
                && frontier[dominated_end].weight <= candidate.weight
            {
                dominated_end += 1;
            }
            frontier.splice(low..dominated_end, std::iter::once(candidate));
        }
    }

    if frontier.is_empty() {
        return HashMap::new();
    }
    let mut selection = HashMap::new();
    let mut node = Some(frontier.last().unwrap().node);
    while let Some(i) = node {
        selection.insert(nodes[i].extraction_index, nodes[i].start);
        node = nodes[i].parent;
    }
    selection
}

#[allow(clippy::too_many_arguments)]
fn apply_monotonic_exact_matches(
    groups: &mut [Vec<Extraction>],
    ordered: &[(usize, usize)],
    source_tokens: &[String],
    tokenized_text: &TokenizedText,
    token_offset: i32,
    char_offset: i32,
    tokenizer_impl: &dyn Tokenizer,
) -> HashSet<(usize, usize)> {
    let token_lists: Vec<Vec<String>> = ordered
        .iter()
        .map(|&(g, i)| tokenize_with_lowercase(&groups[g][i].extraction_text, tokenizer_impl))
        .collect();
    let occ: Vec<Vec<usize>> = token_lists
        .iter()
        .map(|toks| find_token_occurrences(source_tokens, toks))
        .collect();
    let lengths: Vec<usize> = token_lists.iter().map(|t| t.len()).collect();
    let selection = select_monotonic_matches(&occ, &lengths);
    let mut matched = HashSet::new();
    for (extraction_index, start) in selection {
        let (g, i) = ordered[extraction_index];
        let num_tokens = token_lists[extraction_index].len();
        let extraction = &mut groups[g][i];
        extraction.set_token_interval(Some(TokenInterval::new(
            start as i32 + token_offset,
            (start + num_tokens) as i32 + token_offset,
        )));
        let start_token = &tokenized_text.tokens[start];
        let end_token = &tokenized_text.tokens[start + num_tokens - 1];
        extraction.char_interval = Some(CharInterval::new(
            char_offset + start_token.char_interval.start_pos,
            char_offset + end_token.char_interval.end_pos,
        ));
        extraction.alignment_status = Some(AlignmentStatus::MatchExact);
        matched.insert((g, i));
    }
    matched
}

/// Difflib-compatible matching blocks, including the dummy terminator.
fn matching_blocks(a: &[String], b: &[String]) -> Vec<(usize, usize, usize)> {
    let la = a.len();
    let lb = b.len();
    let mut b2j: HashMap<&str, Vec<usize>> = HashMap::new();
    for (j, t) in b.iter().enumerate() {
        b2j.entry(t.as_str()).or_default().push(j);
    }
    let mut queue = vec![(0usize, la, 0usize, lb)];
    let mut matching_blocks = Vec::new();
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = find_longest_match(a, b, &b2j, alo, ahi, blo, bhi);
        if k > 0 {
            matching_blocks.push((i, j, k));
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    matching_blocks.sort_unstable();
    let mut non_adjacent = Vec::new();
    let mut i1 = 0usize;
    let mut j1 = 0usize;
    let mut k1 = 0usize;
    for (i2, j2, k2) in matching_blocks {
        if i1 + k1 == i2 && j1 + k1 == j2 {
            k1 += k2;
        } else {
            if k1 > 0 {
                non_adjacent.push((i1, j1, k1));
            }
            i1 = i2;
            j1 = j2;
            k1 = k2;
        }
    }
    if k1 > 0 {
        non_adjacent.push((i1, j1, k1));
    }
    non_adjacent.push((la, lb, 0));
    non_adjacent
}

#[allow(clippy::needless_range_loop)]
fn find_longest_match(
    a: &[String],
    b: &[String],
    b2j: &HashMap<&str, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let mut besti = alo;
    let mut bestj = blo;
    let mut bestsize = 0usize;
    let mut j2len: HashMap<usize, usize> = HashMap::new();
    for i in alo..ahi {
        let mut newj2len: HashMap<usize, usize> = HashMap::new();
        if let Some(js) = b2j.get(a[i].as_str()) {
            for &j in js {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = j2len.get(&j.wrapping_sub(1)).copied().unwrap_or(0) + 1;
                newj2len.insert(j, k);
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        j2len = newj2len;
    }
    while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
        besti -= 1;
        bestj -= 1;
        bestsize += 1;
    }
    while besti + bestsize < ahi
        && bestj + bestsize < bhi
        && a[besti + bestsize] == b[bestj + bestsize]
    {
        bestsize += 1;
    }
    (besti, bestj, bestsize)
}

fn is_json_int(v: &Value) -> bool {
    json_int(v).is_some()
}

fn json_int(v: &Value) -> Option<i32> {
    v.as_i64()
        .or_else(|| v.as_u64().map(|n| n as i64))
        .and_then(|n| i32::try_from(n).ok())
}

fn value_to_extraction_text(v: &Value) -> Result<String, ResolverError> {
    if let Some(s) = v.as_str() {
        return Ok(s.to_string());
    }
    if let Some(n) = v.as_i64() {
        return Ok(n.to_string());
    }
    if let Some(n) = v.as_u64() {
        return Ok(n.to_string());
    }
    if let Some(n) = v.as_f64() {
        return Ok(n.to_string());
    }
    Err(ResolverError::Schema(
        "Extraction text must be a string, integer, or float.".to_string(),
    ))
}

fn json_attributes(v: &Value) -> Option<indexmap::IndexMap<String, AttributeValue>> {
    let obj = v.as_object()?;
    let mut out = indexmap::IndexMap::new();
    for (k, val) in obj {
        if let Some(s) = val.as_str() {
            out.insert(k.clone(), AttributeValue::String(s.to_string()));
        } else if let Some(arr) = val.as_array() {
            let ss: Vec<String> = arr
                .iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect();
            out.insert(k.clone(), AttributeValue::Strings(ss));
        } else if val.is_null() {
            continue;
        } else {
            out.insert(k.clone(), AttributeValue::String(val.to_string()));
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    fn s(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn best_lcs_span_cases() {
        #[allow(clippy::type_complexity)]
        let cases: &[(&[&str], &[&str], (i32, i32, i32))] = &[
            (&["a", "a", "b"], &["a", "b"], (2, 1, 2)),
            (&["a"], &["a", "a"], (1, 0, 0)),
            (&["x", "a", "b", "c", "y"], &["a", "b", "c"], (3, 1, 3)),
            (&["a", "x", "b", "y", "c"], &["a", "b", "c"], (3, 0, 4)),
            (&["a", "b", "c"], &["x", "y", "z"], (0, -1, -1)),
            (&["a", "b", "c", "a", "b", "c"], &["a", "b", "c"], (3, 0, 2)),
            (&[], &["a"], (0, -1, -1)),
            (&["a"], &[], (0, -1, -1)),
        ];
        for (source, extraction, expect) in cases {
            let r = best_lcs_span(&s(source), &s(extraction));
            assert_eq!(
                (r.matches, r.start, r.end),
                *expect,
                "{source:?} {extraction:?}"
            );
        }
    }

    #[test]
    fn accept_lcs_gate() {
        assert!(accept_lcs_match(
            LcsSpan {
                matches: 3,
                start: 0,
                end: 2
            },
            3,
            0.75,
            1.0 / 3.0
        ));
        assert!(accept_lcs_match(
            LcsSpan {
                matches: 3,
                start: 0,
                end: 8
            },
            3,
            0.75,
            1.0 / 3.0
        ));
        assert!(!accept_lcs_match(
            LcsSpan {
                matches: 3,
                start: 0,
                end: 9
            },
            3,
            0.75,
            1.0 / 3.0
        ));
        assert!(!accept_lcs_match(
            LcsSpan {
                matches: 2,
                start: 0,
                end: 2
            },
            3,
            0.75,
            1.0 / 3.0
        ));
        assert!(accept_lcs_match(
            LcsSpan {
                matches: 3,
                start: 0,
                end: 2
            },
            4,
            0.75,
            1.0 / 3.0
        ));
        assert!(!accept_lcs_match(
            LcsSpan {
                matches: 0,
                start: -1,
                end: -1
            },
            3,
            0.75,
            1.0 / 3.0
        ));
    }

    fn run_fuzzy(
        source: &str,
        extraction_text: &str,
        algorithm: &str,
        token_offset: i32,
        char_offset: i32,
    ) -> Option<Extraction> {
        let tok = RegexTokenizer;
        let tokenized = tok.tokenize(source);
        let source_tokens: Vec<String> = tokenized
            .tokens
            .iter()
            .map(|t| tokenized.token_text(t).to_lowercase())
            .collect();
        let extraction = Extraction::new("entity", extraction_text);
        if algorithm == "lcs" {
            let src_norm: Vec<String> = source_tokens.iter().map(|t| normalize_token(t)).collect();
            WordAligner.lcs_fuzzy_align(
                &extraction,
                &src_norm,
                &tokenized,
                token_offset,
                char_offset,
                &tok,
            )
        } else {
            WordAligner.legacy_fuzzy_align(
                &extraction,
                &source_tokens,
                &tokenized,
                token_offset,
                char_offset,
                &tok,
            )
        }
    }

    #[test]
    fn planted_positive_lcs_and_legacy() {
        let planted = include_str!("../testdata/planted.txt");
        let gapped = include_str!("../testdata/gapped.txt");
        for algo in ["lcs", "legacy"] {
            let r = run_fuzzy(planted, "metformin hydrochloride tablet", algo, 0, 0).unwrap();
            assert_eq!(r.alignment_status, Some(AlignmentStatus::MatchFuzzy));
            assert_eq!(r.token_interval().unwrap(), TokenInterval::new(50, 53));
            let ci = r.char_interval.unwrap();
            assert_eq!((ci.start_pos, ci.end_pos), (Some(451), Some(481)));
            assert_eq!(&planted[451..481], "metformin hydrochloride tablet");

            let r = run_fuzzy(planted, "metformins hydrochlorides tablets", algo, 0, 0).unwrap();
            assert_eq!(r.token_interval().unwrap(), TokenInterval::new(50, 53));

            let r = run_fuzzy(gapped, "metformin hydrochloride tablet", algo, 0, 0).unwrap();
            assert_eq!(r.token_interval().unwrap(), TokenInterval::new(40, 49));
            let ci = r.char_interval.unwrap();
            assert_eq!((ci.start_pos, ci.end_pos), (Some(371), Some(461)));
        }
    }

    #[test]
    fn planted_negative() {
        let base = include_str!("../testdata/base200.txt");
        assert!(run_fuzzy(base, "warfarin coumadin anticoagulant", "lcs", 0, 0).is_none());
        assert!(run_fuzzy(base, "warfarin coumadin anticoagulant", "legacy", 0, 0).is_none());
    }

    #[test]
    fn lcs_repeated_token_min_span() {
        let r = run_fuzzy("a a b", "a b", "lcs", 0, 0).unwrap();
        assert_eq!(r.token_interval().unwrap(), TokenInterval::new(1, 3));
        let ci = r.char_interval.unwrap();
        assert_eq!((ci.start_pos, ci.end_pos), (Some(2), Some(5)));
    }

    #[test]
    fn lcs_density_rejects_sparse() {
        let source = "metformin alpha beta gamma delta epsilon zeta hydrochloride eta tablet";
        assert!(run_fuzzy(source, "metformin hydrochloride tablet", "lcs", 0, 0).is_none());
    }

    #[test]
    fn lcs_accepts_legacy_rejects() {
        let source = "alpha beta gamma";
        let extraction_text = "alpha beta gamma delta";
        let lcs = run_fuzzy(source, extraction_text, "lcs", 0, 0).unwrap();
        assert_eq!(lcs.token_interval().unwrap(), TokenInterval::new(0, 3));
        assert!(run_fuzzy(source, extraction_text, "legacy", 0, 0).is_none());
    }

    #[test]
    fn sparse_max_falls_back_to_dense_submatch() {
        let source =
            "alpha noise noise noise noise noise noise noise noise noise noise beta gamma delta";
        let r = run_fuzzy(source, "alpha beta gamma delta", "lcs", 0, 0).unwrap();
        assert_eq!(r.token_interval().unwrap(), TokenInterval::new(11, 14));
    }

    #[test]
    fn offsets_propagated() {
        let r = run_fuzzy(
            "metformin hydrochloride tablet",
            "metformin hydrochloride tablet",
            "lcs",
            10,
            100,
        )
        .unwrap();
        assert_eq!(r.token_interval().unwrap(), TokenInterval::new(10, 13));
        let ci = r.char_interval.unwrap();
        assert_eq!((ci.start_pos, ci.end_pos), (Some(100), Some(130)));
    }

    fn opts() -> AlignOptions<'static> {
        AlignOptions::default()
    }

    fn align_texts(texts: &[&str], source: &str, exact: &str) -> Vec<Extraction> {
        let groups: Vec<Vec<Extraction>> = texts
            .iter()
            .map(|t| vec![Extraction::new("entity", *t)])
            .collect();
        let options = AlignOptions {
            exact_alignment_algorithm: exact,
            ..opts()
        };
        WordAligner
            .align_groups(groups, source, options)
            .unwrap()
            .into_iter()
            .flatten()
            .collect()
    }

    #[test]
    fn repeated_mentions_successive() {
        let source = "Take aspirin in the morning. Take aspirin at night.";
        let first_start = source.find("aspirin").unwrap() as i32;
        let second_start =
            source[first_start as usize + 1..].find("aspirin").unwrap() as i32 + first_start + 1;
        let aligned = align_texts(&["aspirin", "aspirin"], source, "dp");
        assert_eq!(
            aligned[0].alignment_status,
            Some(AlignmentStatus::MatchExact)
        );
        assert_eq!(
            aligned[1].alignment_status,
            Some(AlignmentStatus::MatchExact)
        );
        assert_eq!(
            aligned[0].char_interval.unwrap().start_pos,
            Some(first_start)
        );
        assert_eq!(
            aligned[1].char_interval.unwrap().start_pos,
            Some(second_start)
        );
    }

    #[test]
    fn long_mention_seed() {
        let source =
            "alpha beta gamma delta was noted. omega followed. Later, alpha beta gamma delta recurred.";
        let aligned = align_texts(&["omega", "alpha beta gamma delta"], source, "dp");
        assert_eq!(
            aligned[0].alignment_status,
            Some(AlignmentStatus::MatchExact)
        );
        assert_eq!(
            aligned[1].alignment_status,
            Some(AlignmentStatus::MatchExact)
        );
        assert_eq!(
            aligned[0].char_interval.unwrap().start_pos,
            Some(source.find("omega").unwrap() as i32)
        );
        assert_eq!(
            aligned[1].char_interval.unwrap().start_pos,
            Some(source.find("alpha beta gamma delta recurred").unwrap() as i32)
        );
    }

    #[test]
    fn difflib_escape_hatch() {
        let source =
            "alpha beta gamma delta was noted. omega followed. Later, alpha beta gamma delta recurred.";
        let aligned = align_texts(&["omega", "alpha beta gamma delta"], source, "difflib");
        assert_eq!(
            aligned[1].alignment_status,
            Some(AlignmentStatus::MatchExact)
        );
        assert_eq!(
            aligned[1].char_interval.unwrap().start_pos,
            Some(source.find("alpha").unwrap() as i32)
        );
        assert_eq!(
            aligned[0].alignment_status,
            Some(AlignmentStatus::MatchFuzzy)
        );
    }

    #[test]
    fn duplicate_single_occurrence() {
        let source = "Only aspirin appears here.";
        let aligned = align_texts(&["aspirin", "aspirin"], source, "dp");
        assert_eq!(
            aligned[0].alignment_status,
            Some(AlignmentStatus::MatchExact)
        );
        assert_eq!(
            aligned[1].alignment_status,
            Some(AlignmentStatus::MatchFuzzy)
        );
        assert_eq!(
            aligned[0].char_interval.unwrap().start_pos,
            aligned[1].char_interval.unwrap().start_pos
        );
    }

    #[test]
    fn exact_upgraded_over_lesser() {
        let source = "Patient has severe heart problems today.";
        let aligned = align_texts(
            &["heart problems", "severe heart problems complications"],
            source,
            "dp",
        );
        assert_eq!(
            aligned[0].alignment_status,
            Some(AlignmentStatus::MatchExact)
        );
        assert_eq!(
            aligned[0].char_interval.unwrap().start_pos,
            Some(source.find("heart").unwrap() as i32)
        );
        assert_eq!(
            aligned[1].alignment_status,
            Some(AlignmentStatus::MatchLesser)
        );
        assert_eq!(
            aligned[1].char_interval.unwrap().start_pos,
            Some(source.find("severe").unwrap() as i32)
        );
    }

    #[test]
    fn dp_offsets() {
        let source = "aspirin taken daily.";
        let groups = vec![vec![Extraction::new("entity", "aspirin")]];
        let options = AlignOptions {
            token_offset: 7,
            char_offset: 100,
            ..opts()
        };
        let aligned = WordAligner.align_groups(groups, source, options).unwrap();
        let e = &aligned[0][0];
        assert_eq!(e.alignment_status, Some(AlignmentStatus::MatchExact));
        assert_eq!(e.token_interval().unwrap().start_index, 7);
        assert_eq!(e.char_interval.unwrap().start_pos, Some(100));
    }

    #[test]
    fn invalid_exact_algorithm() {
        let options = AlignOptions {
            exact_alignment_algorithm: "dP",
            ..opts()
        };
        let err = WordAligner
            .align_groups(
                vec![vec![Extraction::new("e", "aspirin")]],
                "aspirin taken daily.",
                options,
            )
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("Invalid exact_alignment_algorithm"));
    }

    #[test]
    fn invalid_fuzzy_params() {
        let e = Extraction::new("med", "aspirin");
        let bad_thresh = AlignOptions {
            fuzzy_alignment_threshold: 1.5,
            ..opts()
        };
        assert!(WordAligner
            .align_groups(vec![vec![e.clone()]], "patient takes aspirin", bad_thresh)
            .is_err());
        let bad_algo = AlignOptions {
            fuzzy_alignment_algorithm: "bogus",
            ..opts()
        };
        assert!(WordAligner
            .align_groups(vec![vec![e.clone()]], "patient takes aspirin", bad_algo)
            .is_err());
        let disabled = AlignOptions {
            enable_fuzzy_alignment: false,
            fuzzy_alignment_algorithm: "bogus",
            fuzzy_alignment_min_density: -0.1,
            ..opts()
        };
        assert!(WordAligner
            .align_groups(vec![vec![e]], "patient takes aspirin", disabled)
            .is_ok());
    }

    #[test]
    fn resolver_json_fence() {
        let resolver =
            Resolver::from_legacy(true, FormatType::Json, Some(DEFAULT_INDEX_SUFFIX.into()));
        let input = r#"```json
{
  "extractions": [
    {"medication": "Naprosyn", "medication_index": 4, "frequency": "as needed", "frequency_index": 5},
    {"medication": "prednisone", "medication_index": 9, "duration": "for one month", "duration_index": 10}
  ]
}
```"#;
        let xs = resolver.resolve(input, false).unwrap();
        assert_eq!(xs.len(), 4);
        assert_eq!(xs[0].extraction_class, "medication");
        assert_eq!(xs[0].extraction_text, "Naprosyn");
        assert_eq!(xs[0].extraction_index, Some(4));
    }

    #[test]
    fn resolver_yaml_fence() {
        let resolver =
            Resolver::from_legacy(true, FormatType::Yaml, Some(DEFAULT_INDEX_SUFFIX.into()));
        let input = "```yaml\nextractions:\n  - medication: Naprosyn\n    medication_index: 4\n  - duration: for one month\n    duration_index: 10\n```";
        let xs = resolver.resolve(input, false).unwrap();
        assert_eq!(xs[0].extraction_text, "Naprosyn");
        assert_eq!(xs[1].extraction_text, "for one month");
        assert_eq!(xs[1].extraction_index, Some(10));
    }

    #[test]
    fn out_of_order_still_grounds() {
        let source = "aspirin was given before ibuprofen was considered.";
        let aligned = align_texts(&["ibuprofen", "aspirin"], source, "dp");
        for e in &aligned {
            assert!(e.alignment_status.is_some());
            let ci = e.char_interval.unwrap();
            let start = ci.start_pos.unwrap() as usize;
            let end = ci.end_pos.unwrap() as usize;
            assert_eq!(
                source[start..end].to_lowercase(),
                e.extraction_text.to_lowercase()
            );
        }
    }

    #[test]
    fn resolver_integer_extraction() {
        let resolver =
            Resolver::from_legacy(true, FormatType::Json, Some(DEFAULT_INDEX_SUFFIX.into()));
        let input = r#"```json
{"extractions": [{"year": 2006, "year_index": 6}]}
```"#;
        let xs = resolver.resolve(input, false).unwrap();
        assert_eq!(xs[0].extraction_text, "2006");
        assert_eq!(xs[0].extraction_index, Some(6));
    }

    #[test]
    fn resolver_suppress_empty() {
        let resolver =
            Resolver::from_legacy(true, FormatType::Json, Some(DEFAULT_INDEX_SUFFIX.into()));
        assert!(resolver.resolve("```json\n```", true).unwrap().is_empty());
        assert!(resolver.resolve("```json\n```", false).is_err());
    }

    #[test]
    fn resolver_schema_error() {
        let resolver =
            Resolver::from_legacy(true, FormatType::Json, Some(DEFAULT_INDEX_SUFFIX.into()));
        let input = r#"```json
{"extractions": [{"entity": "test", "entity_index": 1, "entity_attributes": "bad"}]}
```"#;
        assert!(resolver.resolve(input, true).unwrap().is_empty());
        assert!(resolver.resolve(input, false).is_err());
    }

    #[test]
    fn resolver_align_chunk() {
        let resolver =
            Resolver::from_legacy(true, FormatType::Json, Some(DEFAULT_INDEX_SUFFIX.into()));
        let text = "This is a sample text with some extractions.";
        let tokenized = tokenize(text);
        let chunk = TokenInterval::new(0, 8);
        let chunk_text = crate::chunking::get_token_interval_text(&tokenized, chunk).unwrap();
        let options = AlignOptions {
            token_offset: chunk.start_index,
            enable_fuzzy_alignment: false,
            ..opts()
        };
        let xs = resolver
            .align(
                vec![
                    Extraction::new("medication", "sample"),
                    Extraction::new("condition", "extractions"),
                ],
                &chunk_text,
                options,
            )
            .unwrap();
        assert_eq!(xs[0].token_interval().unwrap(), TokenInterval::new(3, 4));
        assert_eq!(xs[0].char_interval.unwrap(), CharInterval::new(10, 16));
        assert_eq!(xs[0].alignment_status, Some(AlignmentStatus::MatchExact));
        assert_eq!(xs[1].token_interval().unwrap(), TokenInterval::new(7, 8));
    }

    #[test]
    fn extract_ordered_no_index_suffix() {
        let resolver = Resolver::from_legacy(false, FormatType::Json, None);
        let data = vec![
            serde_json::from_str::<Map<String, Value>>(r#"{"medication":"Aspirin"}"#).unwrap(),
            serde_json::from_str::<Map<String, Value>>(r#"{"medication":"Ibuprofen"}"#).unwrap(),
        ];
        let xs = resolver.extract_ordered_extractions(&data).unwrap();
        assert_eq!(xs[0].extraction_index, Some(1));
        assert_eq!(xs[1].extraction_index, Some(2));
        assert_eq!(xs[0].group_index, Some(0));
        assert_eq!(xs[1].group_index, Some(1));
    }

    #[test]
    fn extract_ordered_index_must_be_int() {
        let resolver =
            Resolver::from_legacy(false, FormatType::Json, Some(DEFAULT_INDEX_SUFFIX.into()));
        let data = vec![serde_json::from_str::<Map<String, Value>>(
            r#"{"medication":"Aspirin","medication_index":"first"}"#,
        )
        .unwrap()];
        assert!(resolver.extract_ordered_extractions(&data).is_err());
    }

    #[test]
    fn clinical_note_duplicates() {
        let source = "\
CLINICAL NOTE
Patient: John Smith    DOB: 03/14/1962
Chief Complaint: Follow-up of hypertension and type 2 diabetes.
History: The patient reports good adherence to lisinopril 20 mg
daily. Blood pressure remains elevated at 148/92 today. Home logs
show fasting values of 130-150 mg/dL on metformin 1000 mg twice
daily. He continues atorvastatin 40 mg nightly for hyperlipidemia.
Assessment and Plan:
1. Hypertension, uncontrolled. Increase lisinopril to 40 mg daily.
2. Type 2 diabetes, above goal. Start empagliflozin 10 mg daily.
3. Hyperlipidemia, stable. Continue atorvastatin.
";
        let texts = [
            "lisinopril",
            "148/92",
            "metformin",
            "atorvastatin",
            "Hypertension",
            "lisinopril",
            "Type 2 diabetes",
            "empagliflozin",
            "Hyperlipidemia",
        ];
        let expected_starts = [
            source.find("lisinopril 20 mg").unwrap(),
            source.find("148/92").unwrap(),
            source.find("metformin").unwrap(),
            source.find("atorvastatin 40 mg").unwrap(),
            source.find("Hypertension, uncontrolled").unwrap(),
            source.find("lisinopril to 40 mg").unwrap(),
            source.find("Type 2 diabetes, above").unwrap(),
            source.find("empagliflozin").unwrap(),
            source.find("Hyperlipidemia, stable").unwrap(),
        ];
        let aligned = align_texts(&texts, source, "dp");
        for (e, start) in aligned.iter().zip(expected_starts) {
            assert_eq!(
                e.alignment_status,
                Some(AlignmentStatus::MatchExact),
                "{}",
                e.extraction_text
            );
            assert_eq!(
                e.char_interval.unwrap().start_pos,
                Some(start as i32),
                "{}",
                e.extraction_text
            );
        }
    }
}
