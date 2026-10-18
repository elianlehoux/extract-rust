// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! HTML visualization of document extractions with interactive playback,
//! slider controls, color-coded class highlights, and sweep-line nested spans.

use std::collections::HashMap;

use indexmap::IndexMap;
use serde::Serialize;

use crate::types::{AnnotatedDocument, AttributeValue, CharInterval, Extraction};

/// Material design inspired palette for extraction classes.
pub const PALETTE: &[&str] = &[
    "#D2E3FC", // Light Blue (Primary Container)
    "#C8E6C9", // Light Green (Tertiary Container)
    "#FEF0C3", // Light Yellow (Primary Color)
    "#F9DEDC", // Light Red (Error Container)
    "#FFDDBE", // Light Orange (Tertiary Container)
    "#EADDFF", // Light Purple (Secondary/Tertiary Container)
    "#C4E9E4", // Light Teal (Teal Container)
    "#FCE4EC", // Light Pink (Pink Container)
    "#E8EAED", // Very Light Grey (Neutral Highlight)
    "#DDE8E8", // Pale Cyan (Cyan Container)
];

/// Embedded CSS stylesheet for extraction visualization.
pub const VISUALIZATION_CSS: &str = r#"<style>
.lx-highlight { position: relative; border-radius:3px; padding:1px 2px;}
.lx-highlight .lx-tooltip {
  visibility: hidden;
  opacity: 0;
  transition: opacity 0.2s ease-in-out;
  background: #333;
  color: #fff;
  text-align: left;
  border-radius: 4px;
  padding: 6px 8px;
  position: absolute;
  z-index: 1000;
  bottom: 125%;
  left: 50%;
  transform: translateX(-50%);
  font-size: 12px;
  max-width: 240px;
  white-space: normal;
  box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}
.lx-highlight:hover .lx-tooltip { visibility: visible; opacity:1; }
.lx-animated-wrapper { max-width: 100%; font-family: Arial, sans-serif; }
.lx-controls {
  background: #fafafa; border: 1px solid #90caf9; border-radius: 8px;
  padding: 12px; margin-bottom: 16px;
}
.lx-button-row {
  display: flex; justify-content: center; gap: 8px; margin-bottom: 12px;
}
.lx-control-btn {
  background: #4285f4; color: white; border: none; border-radius: 4px;
  padding: 8px 16px; cursor: pointer; font-size: 13px; font-weight: 500;
  transition: background-color 0.2s;
}
.lx-control-btn:hover { background: #3367d6; }
.lx-progress-container {
  margin-bottom: 8px;
}
.lx-progress-slider {
  width: 100%; margin: 0; appearance: none; height: 6px;
  background: #ddd; border-radius: 3px; outline: none;
}
.lx-progress-slider::-webkit-slider-thumb {
  appearance: none; width: 18px; height: 18px; background: #4285f4;
  border-radius: 50%; cursor: pointer;
}
.lx-progress-slider::-moz-range-thumb {
  width: 18px; height: 18px; background: #4285f4; border-radius: 50%;
  cursor: pointer; border: none;
}
.lx-status-text {
  text-align: center; font-size: 12px; color: #666; margin-top: 4px;
}
.lx-text-window {
  font-family: monospace; white-space: pre-wrap; border: 1px solid #90caf9;
  padding: 12px; max-height: 260px; overflow-y: auto; margin-bottom: 12px;
  line-height: 1.6;
}
.lx-attributes-panel {
  background: #fafafa; border: 1px solid #90caf9; border-radius: 6px;
  padding: 8px 10px; margin-top: 8px; font-size: 13px;
}
.lx-current-highlight {
  border-bottom: 4px solid #ff4444;
  font-weight: bold;
  animation: lx-pulse 1s ease-in-out;
}
@keyframes lx-pulse {
  0% { text-decoration-color: #ff4444; }
  50% { text-decoration-color: #ff0000; }
  100% { text-decoration-color: #ff4444; }
}
.lx-legend {
  font-size: 12px; margin-bottom: 8px;
  padding-bottom: 8px; border-bottom: 1px solid #e0e0e0;
}
.lx-label {
  display: inline-block;
  padding: 2px 4px;
  border-radius: 3px;
  margin-right: 4px;
  color: #000;
}
.lx-attr-key {
  font-weight: 600;
  color: #1565c0;
  letter-spacing: 0.3px;
}
.lx-attr-value {
  font-weight: 400;
  opacity: 0.85;
  letter-spacing: 0.2px;
}

/* Add optimizations with larger fonts and better readability for GIFs */
.lx-gif-optimized .lx-text-window { font-size: 16px; line-height: 1.8; }
.lx-gif-optimized .lx-attributes-panel { font-size: 15px; }
.lx-gif-optimized .lx-current-highlight { text-decoration-thickness: 4px; }
</style>"#;

/// Configuration options for extraction visualization.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualizeOptions {
    /// Animation playback speed in seconds per step.
    pub animation_speed: f64,
    /// Whether to display the class highlight legend.
    pub show_legend: bool,
    /// Whether to apply larger font and high-contrast styles for GIF export.
    pub gif_optimized: bool,
    /// Context window length in characters around each extraction.
    pub context_chars: usize,
}

impl Default for VisualizeOptions {
    fn default() -> Self {
        Self {
            animation_speed: 1.0,
            show_legend: true,
            gif_optimized: true,
            context_chars: 150,
        }
    }
}

impl VisualizeOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_animation_speed(mut self, speed: f64) -> Self {
        self.animation_speed = speed;
        self
    }

    pub fn with_legend(mut self, show_legend: bool) -> Self {
        self.show_legend = show_legend;
        self
    }

    pub fn with_gif_optimized(mut self, gif_optimized: bool) -> Self {
        self.gif_optimized = gif_optimized;
        self
    }

    pub fn with_context_chars(mut self, context_chars: usize) -> Self {
        self.context_chars = context_chars;
        self
    }
}

/// Visualizes document extractions as an interactive HTML widget.
///
/// If the document contains no valid extractions, returns an HTML snippet stating
/// "No valid extractions to animate."
pub fn visualize(doc: &AnnotatedDocument) -> String {
    visualize_with_options(doc, &VisualizeOptions::default())
}

/// Visualizes document extractions with custom options.
pub fn visualize_with_options(doc: &AnnotatedDocument, options: &VisualizeOptions) -> String {
    let text = match &doc.text {
        Some(t) => t.as_str(),
        None => {
            return format!(
                "{VISUALIZATION_CSS}<div class=\"lx-animated-wrapper\"><p>No valid extractions to animate.</p></div>"
            );
        }
    };

    let valid_extractions = filter_valid_extractions(doc.extractions.as_deref().unwrap_or(&[]));
    if valid_extractions.is_empty() {
        return format!(
            "{VISUALIZATION_CSS}<div class=\"lx-animated-wrapper\"><p>No valid extractions to animate.</p></div>"
        );
    }

    let color_map = assign_colors(&valid_extractions);
    let mut full_html = format!(
        "{VISUALIZATION_CSS}{}",
        build_visualization_html(text, valid_extractions, &color_map, options)
    );

    if options.gif_optimized {
        full_html = full_html.replace(
            "class=\"lx-animated-wrapper\"",
            "class=\"lx-animated-wrapper lx-gif-optimized\"",
        );
    }

    full_html
}

/// Filters extractions to only include those with complete character intervals.
pub fn filter_valid_extractions(extractions: &[Extraction]) -> Vec<&Extraction> {
    extractions
        .iter()
        .filter(|e| {
            if let Some(ci) = e.char_interval {
                if let (Some(s), Some(end)) = (ci.start_pos, ci.end_pos) {
                    return s >= 0 && s < end;
                }
            }
            false
        })
        .collect()
}

/// Assigns a background color from the palette to each distinct extraction class.
pub fn assign_colors(extractions: &[&Extraction]) -> HashMap<String, String> {
    let mut classes: Vec<&str> = extractions
        .iter()
        .filter(|e| e.char_interval.is_some())
        .map(|e| e.extraction_class.as_str())
        .collect();
    classes.sort_unstable();
    classes.dedup();

    let mut color_map = HashMap::new();
    for (i, cls) in classes.into_iter().enumerate() {
        let color = PALETTE[i % PALETTE.len()];
        color_map.insert(cls.to_string(), color.to_string());
    }
    color_map
}

/// Convenience helper to assign colors directly from an extraction slice.
pub fn assign_colors_for_extractions(extractions: &[Extraction]) -> HashMap<String, String> {
    let refs: Vec<&Extraction> = extractions.iter().collect();
    assign_colors(&refs)
}

/// Escapes HTML special characters in text strings.
pub fn escape_html(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#x27;"),
            _ => escaped.push(c),
        }
    }
    escaped
}

/// Formats extraction attributes as a single-line HTML string.
pub fn format_attributes(attributes: Option<&IndexMap<String, AttributeValue>>) -> String {
    let Some(attrs) = attributes else {
        return "{}".to_string();
    };

    let mut attrs_parts = Vec::new();
    for (key, val) in attrs {
        let value_str = match val {
            AttributeValue::String(s) => {
                if s.is_empty() || s == "null" {
                    continue;
                }
                s.clone()
            }
            AttributeValue::Strings(list) => {
                if list.is_empty() {
                    continue;
                }
                list.join(", ")
            }
        };
        attrs_parts.push(format!(
            r#"<span class="lx-attr-key">{}</span>: <span class="lx-attr-value">{}</span>"#,
            escape_html(key),
            escape_html(&value_str)
        ));
    }

    if attrs_parts.is_empty() {
        "{}".to_string()
    } else {
        format!("{{{}}}", attrs_parts.join(", "))
    }
}

/// Builds legend HTML showing extraction classes and their associated colors.
pub fn build_legend_html(color_map: &HashMap<String, String>) -> String {
    if color_map.is_empty() {
        return String::new();
    }
    let mut entries: Vec<(&String, &String)> = color_map.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    let mut legend_items = Vec::new();
    for (class_name, color) in entries {
        legend_items.push(format!(
            r#"<span class="lx-label" style="background-color:{color};">{}</span>"#,
            escape_html(class_name)
        ));
    }
    format!(
        r#"<div class="lx-legend">Highlights Legend: {}</div>"#,
        legend_items.join(" ")
    )
}

#[derive(Debug)]
struct CharIndexer<'a> {
    text: &'a str,
    offsets: Vec<usize>,
}

impl<'a> CharIndexer<'a> {
    fn new(text: &'a str) -> Self {
        let offsets: Vec<usize> = text
            .char_indices()
            .map(|(b, _)| b)
            .chain(std::iter::once(text.len()))
            .collect();
        Self { text, offsets }
    }

    fn total_chars(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn to_byte_offset(&self, char_idx: usize) -> usize {
        if char_idx >= self.offsets.len() {
            self.text.len()
        } else {
            self.offsets[char_idx]
        }
    }

    fn slice_chars(&self, start_char: usize, end_char: usize) -> &'a str {
        if start_char >= end_char {
            return "";
        }
        let start_byte = self.to_byte_offset(start_char);
        let end_byte = self.to_byte_offset(end_char);
        if start_byte >= end_byte {
            return "";
        }
        &self.text[start_byte..end_byte]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TagType {
    End,
    Start,
}

#[derive(Debug, Clone)]
struct SpanPoint<'a> {
    position: usize,
    tag_type: TagType,
    span_idx: usize,
    span_length: usize,
    extraction: &'a Extraction,
    order_added: usize,
}

/// Builds highlighted text with nested `<span>` elements using a sweep-line algorithm.
///
/// Ties at the same position are ordered such that:
/// 1. END tags come before START tags.
/// 2. Among END tags, shorter spans close first.
/// 3. Among START tags, longer spans open first.
pub fn build_highlighted_text(
    text: &str,
    extractions: &[&Extraction],
    color_map: &HashMap<String, String>,
) -> String {
    let indexer = CharIndexer::new(text);
    let mut points = Vec::new();
    let mut order_added = 0;

    for (index, extraction) in extractions.iter().enumerate() {
        let ci = match extraction.char_interval {
            Some(ci) => ci,
            None => continue,
        };
        let (Some(s), Some(e)) = (ci.start_pos, ci.end_pos) else {
            continue;
        };
        if s < 0 || s >= e {
            continue;
        }

        let start_pos = s as usize;
        let end_pos = e as usize;
        let span_length = end_pos - start_pos;

        points.push(SpanPoint {
            position: start_pos,
            tag_type: TagType::Start,
            span_idx: index,
            span_length,
            extraction,
            order_added,
        });
        order_added += 1;

        points.push(SpanPoint {
            position: end_pos,
            tag_type: TagType::End,
            span_idx: index,
            span_length,
            extraction,
            order_added,
        });
        order_added += 1;
    }

    points.sort_by(|a, b| {
        a.position
            .cmp(&b.position)
            .then_with(|| match (a.tag_type, b.tag_type) {
                (TagType::End, TagType::Start) => std::cmp::Ordering::Less,
                (TagType::Start, TagType::End) => std::cmp::Ordering::Greater,
                (TagType::End, TagType::End) => {
                    // Shorter ENDs first (ascending span_length)
                    a.span_length
                        .cmp(&b.span_length)
                        .then_with(|| a.order_added.cmp(&b.order_added))
                }
                (TagType::Start, TagType::Start) => {
                    // Longer STARTs first (descending span_length)
                    b.span_length
                        .cmp(&a.span_length)
                        .then_with(|| a.order_added.cmp(&b.order_added))
                }
            })
    });

    let mut html_parts = Vec::new();
    let mut cursor = 0;

    for point in &points {
        if point.position > cursor {
            html_parts.push(escape_html(indexer.slice_chars(cursor, point.position)));
        }

        match point.tag_type {
            TagType::Start => {
                let color = color_map
                    .get(&point.extraction.extraction_class)
                    .map(|s| s.as_str())
                    .unwrap_or("#ffff8d");
                let highlight_class = if point.span_idx == 0 {
                    " lx-current-highlight"
                } else {
                    ""
                };
                html_parts.push(format!(
                    r#"<span class="lx-highlight{highlight_class}" data-idx="{}" style="background-color:{color};">"#,
                    point.span_idx
                ));
            }
            TagType::End => {
                html_parts.push("</span>".to_string());
            }
        }

        cursor = point.position;
    }

    if cursor < indexer.total_chars() {
        html_parts.push(escape_html(
            indexer.slice_chars(cursor, indexer.total_chars()),
        ));
    }

    html_parts.join("")
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExtractionJsData {
    index: usize,
    class: String,
    text: String,
    color: String,
    start_pos: usize,
    end_pos: usize,
    before_text: String,
    extraction_text: String,
    after_text: String,
    attributes_html: String,
}

fn prepare_extraction_data(
    text: &str,
    extractions: &[&Extraction],
    color_map: &HashMap<String, String>,
    context_chars: usize,
) -> Vec<ExtractionJsData> {
    let indexer = CharIndexer::new(text);
    let total_chars = indexer.total_chars();

    let mut data = Vec::with_capacity(extractions.len());
    for (i, extraction) in extractions.iter().enumerate() {
        let ci = extraction
            .char_interval
            .unwrap_or_else(|| CharInterval::new(0, 0));
        let start_pos = ci.start_pos.unwrap_or(0).max(0) as usize;
        let end_pos = ci.end_pos.unwrap_or(0).max(0) as usize;

        let context_start = start_pos.saturating_sub(context_chars);
        let context_end = (end_pos + context_chars).min(total_chars);

        let before_text = indexer.slice_chars(context_start, start_pos);
        let extraction_text = indexer.slice_chars(start_pos, end_pos);
        let after_text = indexer.slice_chars(end_pos, context_end);

        let color = color_map
            .get(&extraction.extraction_class)
            .map(|s| s.as_str())
            .unwrap_or("#ffff8d");

        let mut attributes_html = format!(
            r#"<div><strong>class:</strong> {}</div>"#,
            escape_html(&extraction.extraction_class)
        );
        attributes_html.push_str(&format!(
            r#"<div><strong>attributes:</strong> {}</div>"#,
            format_attributes(extraction.attributes.as_ref())
        ));

        data.push(ExtractionJsData {
            index: i,
            class: extraction.extraction_class.clone(),
            text: extraction.extraction_text.clone(),
            color: color.to_string(),
            start_pos,
            end_pos,
            before_text: escape_html(before_text),
            extraction_text: escape_html(extraction_text),
            after_text: escape_html(after_text),
            attributes_html,
        });
    }

    data
}

fn build_visualization_html(
    text: &str,
    valid_extractions: Vec<&Extraction>,
    color_map: &HashMap<String, String>,
    options: &VisualizeOptions,
) -> String {
    let mut sorted_extractions = valid_extractions;
    sorted_extractions.sort_by(|a, b| {
        let (s_a, e_a) = (
            a.char_interval.and_then(|ci| ci.start_pos).unwrap_or(0),
            a.char_interval.and_then(|ci| ci.end_pos).unwrap_or(0),
        );
        let (s_b, e_b) = (
            b.char_interval.and_then(|ci| ci.start_pos).unwrap_or(0),
            b.char_interval.and_then(|ci| ci.end_pos).unwrap_or(0),
        );
        let len_a = e_a - s_a;
        let len_b = e_b - s_b;
        s_a.cmp(&s_b).then_with(|| len_b.cmp(&len_a))
    });

    let highlighted_text = build_highlighted_text(text, &sorted_extractions, color_map);
    let extraction_data =
        prepare_extraction_data(text, &sorted_extractions, color_map, options.context_chars);
    let legend_html = if options.show_legend {
        build_legend_html(color_map)
    } else {
        String::new()
    };

    let js_data = serde_json::to_string(&extraction_data).unwrap_or_else(|_| "[]".to_string());
    let js_data = escape_json_for_script(&js_data);

    let first = sorted_extractions[0];
    let (first_start, first_end) = (
        first.char_interval.and_then(|ci| ci.start_pos).unwrap_or(0),
        first.char_interval.and_then(|ci| ci.end_pos).unwrap_or(0),
    );
    let pos_info_str = format!("[{first_start}-{first_end}]");

    let count = sorted_extractions.len();
    let slider_max = count.saturating_sub(1);
    let speed = options.animation_speed;
    let speed_str = if speed.fract() == 0.0 {
        format!("{speed:.1}")
    } else {
        format!("{speed}")
    };

    format!(
        r#"
<div class="lx-animated-wrapper">
  <div class="lx-attributes-panel">
    {legend_html}
    <div id="attributesContainer"></div>
  </div>
  <div class="lx-text-window" id="textWindow">
    {highlighted_text}
  </div>
  <div class="lx-controls">
    <div class="lx-button-row">
      <button class="lx-control-btn" onclick="playPause()">▶️ Play</button>
      <button class="lx-control-btn" onclick="prevExtraction()">⏮ Previous</button>
      <button class="lx-control-btn" onclick="nextExtraction()">⏭ Next</button>
    </div>
    <div class="lx-progress-container">
      <input type="range" id="progressSlider" class="lx-progress-slider"
             min="0" max="{slider_max}" value="0"
             onchange="jumpToExtraction(this.value)">
    </div>
    <div class="lx-status-text">
      Entity <span id="entityInfo">1/{count}</span> |
      Pos <span id="posInfo">{pos_info_str}</span>
    </div>
  </div>
</div>

<script>
  (function() {{
    const extractions = {js_data};
    let currentIndex = 0;
    let isPlaying = false;
    let animationInterval = null;
    let animationSpeed = {speed_str};

    function updateDisplay() {{
      const extraction = extractions[currentIndex];
      if (!extraction) return;

      document.getElementById('attributesContainer').innerHTML = extraction.attributesHtml;
      document.getElementById('entityInfo').textContent = (currentIndex + 1) + '/' + extractions.length;
      document.getElementById('posInfo').textContent = '[' + extraction.startPos + '-' + extraction.endPos + ']';
      document.getElementById('progressSlider').value = currentIndex;

      const playBtn = document.querySelector('.lx-control-btn');
      if (playBtn) playBtn.textContent = isPlaying ? '⏸ Pause' : '▶️ Play';

      const prevHighlight = document.querySelector('.lx-text-window .lx-current-highlight');
      if (prevHighlight) prevHighlight.classList.remove('lx-current-highlight');
      const currentSpan = document.querySelector('.lx-text-window span[data-idx="' + currentIndex + '"]');
      if (currentSpan) {{
        currentSpan.classList.add('lx-current-highlight');
        currentSpan.scrollIntoView({{block: 'center', behavior: 'smooth'}});
      }}
    }}

    function nextExtraction() {{
      currentIndex = (currentIndex + 1) % extractions.length;
      updateDisplay();
    }}

    function prevExtraction() {{
      currentIndex = (currentIndex - 1 + extractions.length) % extractions.length;
      updateDisplay();
    }}

    function jumpToExtraction(index) {{
      currentIndex = parseInt(index);
      updateDisplay();
    }}

    function playPause() {{
      if (isPlaying) {{
        clearInterval(animationInterval);
        isPlaying = false;
      }} else {{
        animationInterval = setInterval(nextExtraction, animationSpeed * 1000);
        isPlaying = true;
      }}
      updateDisplay();
    }}

    window.playPause = playPause;
    window.nextExtraction = nextExtraction;
    window.prevExtraction = prevExtraction;
    window.jumpToExtraction = jumpToExtraction;

    updateDisplay();
  }})();
</script>"#
    )
}

/// Make serialized JSON safe to inline inside a `<script>` element.
///
/// HTML ends a script at `</script` in any letter case, so escaping only the
/// lowercase form let `</SCRIPT>` in the source text break out. Escaping every
/// `<` as `\u003c` is valid JSON (`<` can only appear inside strings) and
/// also neutralises `<!--`. U+2028/U+2029 are escaped for older JS engines.
fn escape_json_for_script(json: &str) -> String {
    json.replace('<', "\\u003c")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assign_colors_basic_assignment() {
        let extractions = vec![
            Extraction::new("CLASS_A", "text_a").with_char_interval(0, 1),
            Extraction::new("CLASS_B", "text_b").with_char_interval(1, 2),
        ];

        let actual = assign_colors_for_extractions(&extractions);
        assert_eq!(actual.get("CLASS_A").map(String::as_str), Some(PALETTE[0]));
        assert_eq!(actual.get("CLASS_B").map(String::as_str), Some(PALETTE[1]));
    }

    #[test]
    fn test_build_highlighted_text_single_span_correct_html() {
        let text = "Hello world";
        let extraction = Extraction::new("GREETING", "Hello").with_char_interval(0, 5);
        let extractions = vec![&extraction];
        let mut color_map = HashMap::new();
        color_map.insert("GREETING".to_string(), "#ff0000".to_string());

        let expected_html = r#"<span class="lx-highlight lx-current-highlight" data-idx="0" style="background-color:#ff0000;">Hello</span> world"#;
        let actual_html = build_highlighted_text(text, &extractions, &color_map);
        assert_eq!(actual_html, expected_html);
    }

    #[test]
    fn test_build_highlighted_text_escapes_html_in_text() {
        let text = "Text with <unsafe> content & ampersand.";
        let mut extraction = Extraction::new("UNSAFE_CLASS", "<unsafe> content & ampersand.")
            .with_char_interval(10, 39);
        let mut attrs = IndexMap::new();
        attrs.insert(
            "detail".to_string(),
            AttributeValue::String("Attribute with <tag> & 'quote'".to_string()),
        );
        extraction = extraction.with_attributes(attrs);

        let extractions = vec![&extraction];
        let mut color_map = HashMap::new();
        color_map.insert("UNSAFE_CLASS".to_string(), "#00ff00".to_string());

        let expected_highlighted_segment = "&lt;unsafe&gt; content &amp; ampersand.";
        let expected_html = format!(
            r#"Text with <span class="lx-highlight lx-current-highlight" data-idx="0" style="background-color:#00ff00;">{expected_highlighted_segment}</span>"#
        );

        let actual_html = build_highlighted_text(text, &extractions, &color_map);
        assert_eq!(actual_html, expected_html);
    }

    #[test]
    fn test_visualize_basic_document_renders_correctly() {
        let doc = AnnotatedDocument::new()
            .with_text("Patient needs Aspirin.")
            .with_extractions(vec![
                Extraction::new("MEDICATION", "Aspirin").with_char_interval(14, 21)
            ]);

        let med_color = PALETTE[0];
        let body_html = format!(
            r#"Patient needs <span class="lx-highlight lx-current-highlight" data-idx="0" style="background-color:{med_color};">Aspirin</span>."#
        );
        let legend_html = format!(
            r#"<div class="lx-legend">Highlights Legend: <span class="lx-label" style="background-color:{med_color};">MEDICATION</span></div>"#
        );

        let actual_html = visualize(&doc);

        assert!(actual_html.contains(VISUALIZATION_CSS));
        assert!(actual_html.contains("lx-animated-wrapper"));
        assert!(actual_html.contains(&body_html));
        assert!(actual_html.contains(&legend_html));
    }

    #[test]
    fn test_visualize_no_extractions_renders_text_and_empty_legend() {
        let doc = AnnotatedDocument::new()
            .with_text("No entities here.")
            .with_extractions(vec![]);

        let expected_html = format!(
            "{VISUALIZATION_CSS}<div class=\"lx-animated-wrapper\"><p>No valid extractions to animate.</p></div>"
        );
        let actual_html = visualize(&doc);
        assert_eq!(actual_html, expected_html);
    }

    #[test]
    fn test_visualize_empty_or_none() {
        let doc = AnnotatedDocument::new();
        let expected_html = format!(
            "{VISUALIZATION_CSS}<div class=\"lx-animated-wrapper\"><p>No valid extractions to animate.</p></div>"
        );
        assert_eq!(visualize(&doc), expected_html);

        let doc_with_ungrounded = AnnotatedDocument::new()
            .with_text("Some text")
            .with_extractions(vec![Extraction::new("CLASS_A", "text")]);
        assert_eq!(visualize(&doc_with_ungrounded), expected_html);
    }

    #[test]
    fn test_nested_spans_sweep_line_ordering() {
        // Test nested spans: outer starts at 0..10, inner starts at 0..5
        let text = "0123456789";
        let outer = Extraction::new("OUTER", "0123456789").with_char_interval(0, 10);
        let inner = Extraction::new("INNER", "01234").with_char_interval(0, 5);
        let extractions = vec![&outer, &inner];

        let mut color_map = HashMap::new();
        color_map.insert("OUTER".to_string(), "#111111".to_string());
        color_map.insert("INNER".to_string(), "#222222".to_string());

        let html = build_highlighted_text(text, &extractions, &color_map);
        // Outer is longer, so it opens first; inner closes at 5, outer closes at 10.
        let expected = r#"<span class="lx-highlight lx-current-highlight" data-idx="0" style="background-color:#111111;"><span class="lx-highlight" data-idx="1" style="background-color:#222222;">01234</span>56789</span>"#;
        assert_eq!(html, expected);

        // Test nested spans ending at same position: outer 0..10, inner 5..10
        let outer = Extraction::new("OUTER", "0123456789").with_char_interval(0, 10);
        let inner = Extraction::new("INNER", "56789").with_char_interval(5, 10);
        let extractions = vec![&outer, &inner];
        let html = build_highlighted_text(text, &extractions, &color_map);
        // Inner is shorter, so at position 10 inner END comes before outer END.
        let expected = r#"<span class="lx-highlight lx-current-highlight" data-idx="0" style="background-color:#111111;">01234<span class="lx-highlight" data-idx="1" style="background-color:#222222;">56789</span></span>"#;
        assert_eq!(html, expected);

        // Test abutting spans: span1 0..5, span2 5..10
        let s1 = Extraction::new("S1", "01234").with_char_interval(0, 5);
        let s2 = Extraction::new("S2", "56789").with_char_interval(5, 10);
        let extractions = vec![&s1, &s2];
        let mut color_map = HashMap::new();
        color_map.insert("S1".to_string(), "#111111".to_string());
        color_map.insert("S2".to_string(), "#222222".to_string());
        let html = build_highlighted_text(text, &extractions, &color_map);
        // At position 5, S1 END comes before S2 START.
        let expected = r#"<span class="lx-highlight lx-current-highlight" data-idx="0" style="background-color:#111111;">01234</span><span class="lx-highlight" data-idx="1" style="background-color:#222222;">56789</span>"#;
        assert_eq!(html, expected);
    }

    #[test]
    fn test_format_attributes() {
        assert_eq!(format_attributes(None), "{}");

        let mut attrs = IndexMap::new();
        assert_eq!(format_attributes(Some(&attrs)), "{}");

        attrs.insert(
            "empty_str".to_string(),
            AttributeValue::String("".to_string()),
        );
        attrs.insert(
            "null_str".to_string(),
            AttributeValue::String("null".to_string()),
        );
        attrs.insert("empty_list".to_string(), AttributeValue::Strings(vec![]));
        assert_eq!(format_attributes(Some(&attrs)), "{}");

        attrs.insert(
            "dosage".to_string(),
            AttributeValue::String("10mg".to_string()),
        );
        attrs.insert(
            "tags".to_string(),
            AttributeValue::Strings(vec!["daily".to_string(), "oral".to_string()]),
        );

        let formatted = format_attributes(Some(&attrs));
        assert!(formatted.starts_with('{') && formatted.ends_with('}'));
        assert!(formatted.contains(
            r#"<span class="lx-attr-key">dosage</span>: <span class="lx-attr-value">10mg</span>"#
        ));
        assert!(formatted.contains(r#"<span class="lx-attr-key">tags</span>: <span class="lx-attr-value">daily, oral</span>"#));
    }

    #[test]
    fn test_unicode_character_indexing() {
        let text = "Héllo, 🌍 world!";
        // "Héllo" is 5 chars (6 bytes in utf-8)
        // ", " is 2 chars
        // "🌍" is 1 char (4 bytes in utf-8)
        // " world!" is 7 chars
        // Total chars: 5 + 2 + 1 + 7 = 15 chars.
        let earth = Extraction::new("EMOJI", "🌍").with_char_interval(7, 8);
        let extractions = vec![&earth];
        let mut color_map = HashMap::new();
        color_map.insert("EMOJI".to_string(), "#abcdef".to_string());

        let html = build_highlighted_text(text, &extractions, &color_map);
        let expected = r#"Héllo, <span class="lx-highlight lx-current-highlight" data-idx="0" style="background-color:#abcdef;">🌍</span> world!"#;
        assert_eq!(html, expected);
    }

    #[test]
    fn test_visualize_options_customization() {
        let doc = AnnotatedDocument::new()
            .with_text("Alpha Beta")
            .with_extractions(vec![
                Extraction::new("GREEK", "Alpha").with_char_interval(0, 5)
            ]);

        let opts_no_gif = VisualizeOptions::new()
            .with_gif_optimized(false)
            .with_legend(false)
            .with_animation_speed(2.5);
        let html = visualize_with_options(&doc, &opts_no_gif);
        assert!(!html.contains("class=\"lx-animated-wrapper lx-gif-optimized\""));
        assert!(html.contains("class=\"lx-animated-wrapper\""));
        assert!(!html.contains("Highlights Legend:"));
        assert!(html.contains("animationSpeed = 2.5;"));

        let opts_gif = VisualizeOptions::new().with_gif_optimized(true);
        let html_gif = visualize_with_options(&doc, &opts_gif);
        assert!(html_gif.contains("class=\"lx-animated-wrapper lx-gif-optimized\""));
    }

    #[test]
    fn test_palette_cycling() {
        // Create 12 different extraction classes (more than 10 colors in PALETTE)
        let mut extractions = Vec::new();
        for i in 0..12 {
            let class_name = format!("CLASS_{i:02}");
            extractions
                .push(Extraction::new(class_name, "val").with_char_interval(i * 5, (i + 1) * 5));
        }

        let color_map = assign_colors_for_extractions(&extractions);
        assert_eq!(color_map.len(), 12);
        // Classes are sorted alphabetically: CLASS_00..CLASS_11
        // CLASS_00 should get PALETTE[0], CLASS_10 should get PALETTE[10 % 10 = 0]
        assert_eq!(color_map.get("CLASS_00").unwrap(), PALETTE[0]);
        assert_eq!(color_map.get("CLASS_10").unwrap(), PALETTE[0]);
        assert_eq!(color_map.get("CLASS_11").unwrap(), PALETTE[1]);
    }

    #[test]
    fn test_filter_ungrounded_and_invalid_spans() {
        let extractions = vec![
            Extraction::new("C1", "valid").with_char_interval(0, 5),
            Extraction::new("C2", "ungrounded"),
            Extraction::new("C3", "zero_len").with_char_interval(5, 5),
            Extraction::new("C4", "inverted").with_char_interval(10, 5),
            Extraction::new("C5", "negative").with_char_interval(-3, 2),
            Extraction::new("C6", "valid_2").with_char_interval(6, 12),
        ];

        let filtered = filter_valid_extractions(&extractions);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].extraction_class, "C1");
        assert_eq!(filtered[1].extraction_class, "C6");
    }

    #[test]
    fn test_identical_span_nesting() {
        // Two extractions covering the exact same span
        let text = "Important text";
        let e1 = Extraction::new("TAG1", "Important").with_char_interval(0, 9);
        let e2 = Extraction::new("TAG2", "Important").with_char_interval(0, 9);
        let extractions = vec![&e1, &e2];

        let mut color_map = HashMap::new();
        color_map.insert("TAG1".to_string(), "#111111".to_string());
        color_map.insert("TAG2".to_string(), "#222222".to_string());

        let html = build_highlighted_text(text, &extractions, &color_map);
        // Both spans open at 0 and close at 9. Span 0 opens first, span 1 opens second.
        // At 9, both close.
        let expected = r#"<span class="lx-highlight lx-current-highlight" data-idx="0" style="background-color:#111111;"><span class="lx-highlight" data-idx="1" style="background-color:#222222;">Important</span></span> text"#;
        assert_eq!(html, expected);
    }

    #[test]
    fn test_reexported_visualize() {
        let doc = AnnotatedDocument::new().with_text("Sample");
        let output = crate::visualize(&doc);
        assert!(output.contains("No valid extractions to animate."));
    }

    #[test]
    fn script_escape_is_case_insensitive() {
        for closing in ["</script>", "</SCRIPT>", "</ScRiPt >", "<!--"] {
            let text = format!("Dose {closing}<img src=x onerror=alert(1)> Aspirin.");
            let start = text.find("Aspirin").unwrap() as i32;
            let doc = AnnotatedDocument::new()
                .with_text(text.clone())
                .with_extractions(vec![
                    Extraction::new("MEDICATION", "Aspirin").with_char_interval(start, start + 7)
                ]);
            let html = visualize(&doc).to_ascii_lowercase();
            // Only the visualizer's own <script> blocks may close.
            let opens = html.matches("<script").count();
            let closes = html.matches("</script").count();
            assert_eq!(opens, closes, "{closing}: {html}");
            assert!(!html.contains("<img src=x"), "{closing}");
        }
    }

    #[test]
    fn escaped_json_still_parses() {
        let raw = serde_json::json!({"t": "a </SCRIPT> b \u{2028}"}).to_string();
        let escaped = escape_json_for_script(&raw);
        assert!(!escaped.contains('<'));
        let back: serde_json::Value = serde_json::from_str(&escaped).unwrap();
        assert_eq!(back["t"], "a </SCRIPT> b \u{2028}");
    }
}
