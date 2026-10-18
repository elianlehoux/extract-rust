//! Evaluation runner for the `evals/` set.
//!
//! ```text
//! # run a model on evals/gold.jsonl, save predictions, print scores
//! cargo run --release -p extract-cli --example eval -- run -m gemma2:2b \
//!     --out evals/results/gemma2-2b.rust.jsonl
//!
//! # score an existing predictions file (e.g. from the Python baseline)
//! cargo run --release -p extract-cli --example eval -- score \
//!     --pred evals/results/gemma2-2b.python.jsonl
//! ```
//!
//! Metrics (micro-averaged over all documents):
//! - entity P/R/F1: a prediction is correct when its class matches a gold
//!   entity and its text matches after case/whitespace normalisation;
//! - grounded P/R/F1: class matches and the char interval equals the gold one;
//! - alignment accuracy: share of correct entities whose char interval equals
//!   the gold interval (does the aligner point at the right place?);
//! - grounding rate: share of predictions that received any char interval.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};
use extract_core::extract::{extract, ExtractInput, ExtractOptions, ExtractResult};
use extract_core::prompt_validation::PromptValidationLevel;
use extract_core::types::{ExampleData, FormatType};
use extract_providers::{create_model, resolve_provider, ModelOptions, ProviderKind};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a model over the gold set and score it.
    Run {
        #[arg(short = 'm', long)]
        model: String,
        #[arg(long)]
        provider: Option<String>,
        #[arg(long)]
        base_url: Option<String>,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long, default_value = "evals/gold.jsonl")]
        gold: PathBuf,
        #[arg(long, default_value = "evals/examples")]
        examples_dir: PathBuf,
        /// Where to write predictions (JSONL).
        #[arg(long)]
        out: PathBuf,
        /// Where to write the score summary (JSON). Defaults to <out>.summary.json.
        #[arg(long)]
        summary: Option<PathBuf>,
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
    },
    /// Score a predictions file against the gold set.
    Score {
        #[arg(long)]
        pred: PathBuf,
        #[arg(long, default_value = "evals/gold.jsonl")]
        gold: PathBuf,
        #[arg(long)]
        summary: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Span {
    extraction_class: String,
    extraction_text: String,
    #[serde(default)]
    char_interval: Option<Interval>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
struct Interval {
    start_pos: Option<i64>,
    end_pos: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct GoldDoc {
    id: String,
    domain: String,
    lang: String,
    text: String,
    extractions: Vec<Span>,
}

#[derive(Debug, Deserialize, Serialize)]
struct PredDoc {
    id: String,
    #[serde(default)]
    latency_ms: Option<f64>,
    #[serde(default)]
    error: Option<String>,
    extractions: Vec<Span>,
}

#[derive(Deserialize)]
struct DomainSpec {
    prompt: String,
    examples: Vec<ExampleData>,
}

#[derive(Default, Clone, Copy)]
struct Counts {
    pred: usize,
    gold: usize,
    tp: usize,
    grounded_tp: usize,
    with_interval: usize,
}

impl Counts {
    fn add(&mut self, o: Counts) {
        self.pred += o.pred;
        self.gold += o.gold;
        self.tp += o.tp;
        self.grounded_tp += o.grounded_tp;
        self.with_interval += o.with_interval;
    }
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        a as f64 / b as f64
    }
}

fn f1(p: f64, r: f64) -> f64 {
    if p + r == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

fn norm(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn interval_eq(a: &Option<Interval>, b: &Option<Interval>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.start_pos.is_some() && a == b,
        _ => false,
    }
}

fn has_interval(s: &Span) -> bool {
    s.char_interval
        .map(|i| i.start_pos.is_some() && i.end_pos.is_some())
        .unwrap_or(false)
}

/// Score one document. Greedy one-to-one matching; when several gold
/// entities share class and text, the one with the same interval wins.
fn score_doc(gold: &[Span], pred: &[Span]) -> Counts {
    let mut c = Counts {
        pred: pred.len(),
        gold: gold.len(),
        with_interval: pred.iter().filter(|p| has_interval(p)).count(),
        ..Counts::default()
    };
    let mut used = vec![false; gold.len()];
    for p in pred {
        let candidates: Vec<usize> = (0..gold.len())
            .filter(|&i| {
                !used[i]
                    && gold[i].extraction_class == p.extraction_class
                    && norm(&gold[i].extraction_text) == norm(&p.extraction_text)
            })
            .collect();
        let pick = candidates
            .iter()
            .copied()
            .find(|&i| interval_eq(&gold[i].char_interval, &p.char_interval))
            .or_else(|| candidates.first().copied());
        if let Some(i) = pick {
            used[i] = true;
            c.tp += 1;
            if interval_eq(&gold[i].char_interval, &p.char_interval) {
                c.grounded_tp += 1;
            }
        }
    }
    c
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(
    path: &Path,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?,
        );
    }
    Ok(out)
}

fn summarize(gold: &[GoldDoc], preds: &[PredDoc]) -> Value {
    let by_id: BTreeMap<&str, &PredDoc> = preds.iter().map(|p| (p.id.as_str(), p)).collect();
    let mut total = Counts::default();
    let mut by_domain: BTreeMap<String, Counts> = BTreeMap::new();
    let mut by_lang: BTreeMap<String, Counts> = BTreeMap::new();
    let mut latency = 0.0;
    let mut errors = 0;
    for g in gold {
        let p = by_id.get(g.id.as_str());
        let empty = Vec::new();
        let pred = p.map(|p| &p.extractions).unwrap_or(&empty);
        if p.map(|p| p.error.is_some()).unwrap_or(true) {
            errors += 1;
        }
        latency += p.and_then(|p| p.latency_ms).unwrap_or(0.0);
        let c = score_doc(&g.extractions, pred);
        total.add(c);
        by_domain.entry(g.domain.clone()).or_default().add(c);
        by_lang.entry(g.lang.clone()).or_default().add(c);
    }
    let block = |c: &Counts| {
        let p = ratio(c.tp, c.pred);
        let r = ratio(c.tp, c.gold);
        let gp = ratio(c.grounded_tp, c.pred);
        let gr = ratio(c.grounded_tp, c.gold);
        json!({
            "pred": c.pred, "gold": c.gold, "tp": c.tp, "grounded_tp": c.grounded_tp,
            "precision": p, "recall": r, "f1": f1(p, r),
            "grounded_precision": gp, "grounded_recall": gr, "grounded_f1": f1(gp, gr),
            "alignment_accuracy": ratio(c.grounded_tp, c.tp),
            "grounding_rate": ratio(c.with_interval, c.pred),
        })
    };
    json!({
        "docs": gold.len(),
        "docs_with_errors": errors,
        "total_latency_ms": latency,
        "overall": block(&total),
        "by_domain": by_domain.iter().map(|(k, v)| (k.clone(), block(v))).collect::<serde_json::Map<_, _>>(),
        "by_lang": by_lang.iter().map(|(k, v)| (k.clone(), block(v))).collect::<serde_json::Map<_, _>>(),
    })
}

fn print_summary(s: &Value) {
    let row = |name: &str, b: &Value| {
        println!(
            "| {name:<14} | {:>4} | {:>4} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
            b["gold"],
            b["pred"],
            b["precision"].as_f64().unwrap(),
            b["recall"].as_f64().unwrap(),
            b["f1"].as_f64().unwrap(),
            b["grounded_f1"].as_f64().unwrap(),
            b["alignment_accuracy"].as_f64().unwrap(),
            b["grounding_rate"].as_f64().unwrap(),
        )
    };
    println!("| slice          | gold | pred | P     | R     | F1    | grounded F1 | align acc | grounding rate |");
    println!("|----------------|------|------|-------|-------|-------|-------|-------|-------|");
    row("overall", &s["overall"]);
    for (k, v) in s["by_domain"].as_object().unwrap() {
        row(k, v);
    }
    for (k, v) in s["by_lang"].as_object().unwrap() {
        row(&format!("lang={k}"), v);
    }
    println!(
        "docs={} errors={} total_latency={:.1}s",
        s["docs"],
        s["docs_with_errors"],
        s["total_latency_ms"].as_f64().unwrap() / 1000.0
    );
}

fn expects_fences(kind: Option<ProviderKind>, format_type: FormatType) -> bool {
    !(kind == Some(ProviderKind::Ollama) && format_type == FormatType::Json)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Score {
            pred,
            gold,
            summary,
        } => {
            let gold: Vec<GoldDoc> = read_jsonl(&gold)?;
            let preds: Vec<PredDoc> = read_jsonl(&pred)?;
            let s = summarize(&gold, &preds);
            print_summary(&s);
            let path = summary.unwrap_or_else(|| pred.with_extension("summary.json"));
            fs::write(path, serde_json::to_string_pretty(&s)?)?;
        }
        Cmd::Run {
            model,
            provider,
            base_url,
            api_key,
            gold,
            examples_dir,
            out,
            summary,
            temperature,
        } => {
            let gold_docs: Vec<GoldDoc> = read_jsonl(&gold)?;
            let provider = provider.as_deref().map(ProviderKind::parse).transpose()?;
            let kind = provider.or_else(|| resolve_provider(&model).ok());
            let mut lm = create_model(
                &model,
                ModelOptions {
                    provider,
                    api_key,
                    base_url,
                    format_type: Some(FormatType::Json),
                    temperature: Some(temperature),
                    timeout_secs: Some(300),
                    max_workers: Some(1),
                },
            )?;
            let mut specs: BTreeMap<String, DomainSpec> = BTreeMap::new();
            let mut file = fs::File::create(&out)?;
            for g in &gold_docs {
                if !specs.contains_key(&g.domain) {
                    let path = examples_dir.join(format!("{}.json", g.domain));
                    specs.insert(
                        g.domain.clone(),
                        serde_json::from_str(&fs::read_to_string(path)?)?,
                    );
                }
                let spec = &specs[&g.domain];
                let options = ExtractOptions {
                    format_type: FormatType::Json,
                    fence_output: expects_fences(kind, FormatType::Json),
                    prompt_validation_level: PromptValidationLevel::Off,
                    ..ExtractOptions::default()
                };
                let t0 = Instant::now();
                let res = extract(
                    ExtractInput::Text(g.text.clone()),
                    &spec.prompt,
                    &spec.examples,
                    lm.as_mut(),
                    options,
                );
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let (extractions, error) = match res {
                    Ok(ExtractResult::Document(d)) => (
                        d.extractions
                            .unwrap_or_default()
                            .into_iter()
                            .map(|e| Span {
                                extraction_class: e.extraction_class,
                                extraction_text: e.extraction_text,
                                char_interval: e.char_interval.map(|c| Interval {
                                    start_pos: c.start_pos.map(i64::from),
                                    end_pos: c.end_pos.map(i64::from),
                                }),
                            })
                            .collect(),
                        None,
                    ),
                    Ok(ExtractResult::Documents(_)) => {
                        (Vec::new(), Some("unexpected multi-doc".into()))
                    }
                    Err(e) => (Vec::new(), Some(e.to_string())),
                };
                eprintln!(
                    "{:<9} {:>7.0} ms  {} extractions{}",
                    g.id,
                    ms,
                    extractions.len(),
                    error
                        .as_deref()
                        .map(|e| format!("  ERROR: {e}"))
                        .unwrap_or_default()
                );
                let rec = PredDoc {
                    id: g.id.clone(),
                    latency_ms: Some(ms),
                    error,
                    extractions,
                };
                writeln!(file, "{}", serde_json::to_string(&rec)?)?;
            }
            drop(file);
            let preds: Vec<PredDoc> = read_jsonl(&out)?;
            let mut s = summarize(&gold_docs, &preds);
            s["model"] = json!(model);
            s["implementation"] = json!("extract-rust");
            print_summary(&s);
            let path = summary.unwrap_or_else(|| out.with_extension("summary.json"));
            fs::write(path, serde_json::to_string_pretty(&s)?)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(c: &str, t: &str, s: i64, e: i64) -> Span {
        Span {
            extraction_class: c.into(),
            extraction_text: t.into(),
            char_interval: Some(Interval {
                start_pos: Some(s),
                end_pos: Some(e),
            }),
        }
    }

    #[test]
    fn scoring_counts_text_and_grounding_separately() {
        let gold = vec![
            span("person", "Ada", 0, 3),
            span("person", "Ada", 10, 13),
            span("org", "MIT", 20, 23),
        ];
        let pred = vec![
            span("person", "ada", 10, 13),   // right text, right place
            span("person", "Ada", 4, 7),     // right text, wrong place
            span("location", "MIT", 20, 23), // wrong class
        ];
        let c = score_doc(&gold, &pred);
        assert_eq!((c.tp, c.grounded_tp, c.pred, c.gold), (2, 1, 3, 3));
    }
}
