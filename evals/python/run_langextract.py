#!/usr/bin/env python3
"""Baseline: run upstream Google LangExtract on the same eval set.

Writes predictions in the same JSONL format as the Rust runner so both are
scored by the same code:

    pip install langextract
    python evals/python/run_langextract.py -m gemma2:2b \
        --out evals/results/gemma2-2b.python.jsonl
    cargo run --release -p extract-cli --example eval -- score \
        --pred evals/results/gemma2-2b.python.jsonl

Settings mirror the Rust runner: temperature 0, one worker, JSON output via
Ollama's format=json, no fences, prompt validation off.
"""
import argparse
import json
import pathlib
import time

import langextract as lx
from langextract import prompt_validation as pv


def load_spec(path: pathlib.Path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    examples = [
        lx.data.ExampleData(
            text=ex["text"],
            extractions=[
                lx.data.Extraction(
                    extraction_class=e["extraction_class"],
                    extraction_text=e["extraction_text"],
                    attributes=e.get("attributes"),
                )
                for e in ex["extractions"]
            ],
        )
        for ex in spec["examples"]
    ]
    return spec["prompt"], examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--model-url", default="http://localhost:11434")
    ap.add_argument("--gold", default="evals/gold.jsonl")
    ap.add_argument("--examples-dir", default="evals/examples")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    specs = {}
    with open(args.gold, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            doc = json.loads(line)
            dom = doc["domain"]
            if dom not in specs:
                specs[dom] = load_spec(pathlib.Path(args.examples_dir) / f"{dom}.json")
            prompt, examples = specs[dom]
            t0 = time.perf_counter()
            error = None
            exts = []
            try:
                res = lx.extract(
                    text_or_documents=doc["text"],
                    prompt_description=prompt,
                    examples=examples,
                    model_id=args.model,
                    model_url=args.model_url,
                    temperature=0.0,
                    fence_output=False,
                    use_schema_constraints=True,
                    max_workers=1,
                    show_progress=False,
                    prompt_validation_level=pv.PromptValidationLevel.OFF,
                )
                for e in res.extractions or []:
                    ci = e.char_interval
                    exts.append({
                        "extraction_class": e.extraction_class,
                        "extraction_text": e.extraction_text,
                        "char_interval": None if ci is None else {
                            "start_pos": ci.start_pos, "end_pos": ci.end_pos},
                    })
            except Exception as exc:  # keep going; the error is recorded
                error = f"{type(exc).__name__}: {exc}"
            ms = (time.perf_counter() - t0) * 1000
            print(f"{doc['id']:<9} {ms:7.0f} ms  {len(exts)} extractions"
                  + (f"  ERROR: {error}" if error else ""), flush=True)
            out.write(json.dumps({"id": doc["id"], "latency_ms": ms, "error": error,
                                  "extractions": exts}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
