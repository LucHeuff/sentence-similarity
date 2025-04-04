import cProfile
import warnings
from pstats import Stats
from time import perf_counter

import numpy as np
import polars as pl
from sentence_similarity import sentence_similarity

warnings.simplefilter("ignore", DeprecationWarning)


bench = pl.scan_csv("omschrijving.csv").unique().collect()
sentences = bench["omschrijving"].to_list()

# Benchmarking

N = 10

times = []

for _ in range(N):
    start = perf_counter()
    sentence_similarity(sentences)
    end = perf_counter()
    times.append(end - start)

result = np.asarray(times)

print(" ---- BENCHMARK ----")  # noqa: T201
print(f"Average time: {result.mean():.3f} ± {result.std():.3f} s\n")  # noqa: T201

print(" ---- PROFILE ----")  # noqa: T201

# Profiling
PROFILE = True

if PROFILE:
    with cProfile.Profile() as profiler:
        sentence_similarity(sentences)
        Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)
