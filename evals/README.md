# Evals

A small, hand-annotated set to measure extraction quality and grounding.

- `gold.jsonl`: 20 texts (13 English, 7 Spanish), 86 gold
  entities with character offsets. Generated from the annotations in
  `tools/make_gold.py`, which checks that every span exists in its text.
- `examples/<domain>.json`: the prompt and few-shot examples used for each
  domain. None of the few-shot texts appear in the gold set.
- `results/`: predictions (`*.jsonl`) and scores (`*.summary.json`, `*.table.md`)
  from real runs.

| domain | docs | classes |
|---|---|---|
| general | 6 (3 en, 3 es) | person, organization, location |
| medical | 5 (3 en, 2 es) | medication, dosage, frequency |
| metar | 4 (en) | station, wind, visibility, clouds, temperature, pressure |
| flight_report | 5 (3 en, 2 es) | flight_number, aircraft, airport |

## Run

```bash
# local model through Ollama
cargo run --release -p extract-cli --example eval -- run -m gemma2:2b \
    --out evals/results/gemma2-2b.rust.jsonl

# score any predictions file (same JSONL format)
cargo run --release -p extract-cli --example eval -- score \
    --pred evals/results/gemma2-2b.python.jsonl

# upstream Python LangExtract baseline on the same set
pip install langextract
python evals/python/run_langextract.py -m gemma2:2b \
    --out evals/results/gemma2-2b.python.jsonl
```

## Metrics

- **Entity P/R/F1**: a prediction counts when its class matches a gold entity and
  its text matches after lowercasing and collapsing whitespace (one-to-one).
- **Grounded F1**: same, but the predicted `char_interval` must equal the gold one.
- **Alignment accuracy**: of the correct entities, the share whose offsets equal
  the gold offsets. This isolates the aligner from the model.
- **Grounding rate**: share of predictions that received any offsets.

The set is small (86 entities), so one entity moves F1 by about one point.
Treat the numbers as a smoke test and a regression guard, not a benchmark.

## Offline benchmark (no LLM)

`bench/` measures the library's own work (chunking, prompting, parsing,
alignment) with a deterministic fake model, in Rust and in Python:

```bash
python3 evals/bench/gen_corpus.py 10000 > /tmp/corpus.txt   # ~700 KB
cargo run --release -p extract-cli --example bench_offline -- /tmp/corpus.txt /tmp/rs.json
python evals/bench/fake_model_py.py /tmp/corpus.txt /tmp/py.json
```
