#!/usr/bin/env python3
"""Offline benchmark for upstream LangExtract with a deterministic fake model.

The fake model reads the chunk after the last "Q: " in the prompt and returns
every "Firstname Lastname" pair found by a regex, so the measured time is the
library's own work (chunking, prompting, parsing, alignment), not an LLM.

python3 evals/bench/fake_model_py.py /tmp/corpus.txt /tmp/py_out.json
"""
import json
import re
import sys
import time

import langextract as lx
from langextract import prompt_validation as pv
from langextract.core import base_model, types

NAME = re.compile(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+ [A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\b")


def fake_answer(prompt: str) -> str:
    q = prompt.rfind("Q: ")
    a = prompt.rfind("\nA:")
    chunk = prompt[q + 3:a] if q >= 0 and a > q else ""
    return json.dumps({"extractions": [{"person": m} for m in NAME.findall(chunk)]}, ensure_ascii=False)


class FakeModel(base_model.BaseLanguageModel):
    def infer(self, batch_prompts, **kwargs):
        for p in batch_prompts:
            yield [types.ScoredOutput(score=1.0, output=fake_answer(p))]


def main():
    text = open(sys.argv[1], encoding="utf-8").read()
    examples = [lx.data.ExampleData(
        text="Alan Turing worked with Grace Hopper.",
        extractions=[lx.data.Extraction("person", "Alan Turing"),
                     lx.data.Extraction("person", "Grace Hopper")])]
    t0 = time.perf_counter()
    doc = lx.extract(text_or_documents=text, prompt_description="Extract people.",
                     examples=examples, model=FakeModel(), fence_output=False,
                     use_schema_constraints=False, max_char_buffer=1000, batch_length=10,
                     max_workers=1, show_progress=False,
                     prompt_validation_level=pv.PromptValidationLevel.OFF)
    dt = time.perf_counter() - t0
    exts = [{"extraction_class": e.extraction_class, "extraction_text": e.extraction_text,
             "start": e.char_interval.start_pos if e.char_interval else None,
             "end": e.char_interval.end_pos if e.char_interval else None,
             "status": e.alignment_status.value if e.alignment_status else None}
            for e in doc.extractions]
    json.dump(exts, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
    print(f"python extract(): {dt:.3f}s, {len(exts)} extractions")


main()
