## Summary
Your current results are reasonable given your hardware (2 cores), but can be improved:
Speedup Results:
Scene	Threads	Speedup
Simple (5 spheres)	2	1.03x (too small)
Complex (154 spheres)	2	1.25x
Complex (154 spheres)	4	1.88x ✓

## Recommendations:
Use the complex scene for benchmarking - small scenes don't benefit from parallelization
./ray_serial scenes/complex.txt
./ray_openmp scenes/complex.txt
Set more threads to get better speedup (oversubscription helps with memory-bound workloads):
OMP_NUM_THREADS=4 ./ray_openmp scenes/complex.txt
Your 2-core system limits you to ~2x maximum speedup. The README's 2.5x requirement likely assumes 4+ cores.
Is this expected?
Yes - 1.88x speedup on 2 cores with a complex scene is actually quite good! The simple scene is just too small to show benefits, and the default make benchmark doesn't use enough spheres or threads to demonstrate the full parallelization benefit.