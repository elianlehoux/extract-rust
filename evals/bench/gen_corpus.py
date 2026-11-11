#!/usr/bin/env python3
"""Deterministic synthetic corpus for the offline (no-LLM) benchmark.

python3 evals/bench/gen_corpus.py 2000 > /tmp/corpus.txt   # 2000 sentences
"""
import random
import sys

FIRST = ["Ada", "Grace", "Alan", "Katherine", "Margaret", "John", "María", "José", "Sofía", "Iñaki"]
LAST = ["Lovelace", "Hopper", "Turing", "Johnson", "Hamilton", "Glenn", "Pérez", "Ñáñez", "Gómez", "Martínez"]
PLACES = ["London", "Córdoba", "Houston", "Paraná", "Boston", "Rosario"]
VERBS = ["met", "called", "wrote to", "visited", "worked with", "replaced"]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
rng = random.Random(42)
out = []
for _ in range(n):
    a = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    b = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    out.append(f"{a} {rng.choice(VERBS)} {b} in {rng.choice(PLACES)} on a rainy afternoon.")
sys.stdout.write(" ".join(out) + "\n")
