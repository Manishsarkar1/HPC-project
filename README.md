# Distributed Graph Processing with MPI4Py

This project evaluates MPI4Py implementations of Breadth-First Search, PageRank, and connected components on a distributed graph. It uses one-dimensional vertex partitioning and native MPI collectives, including `Alltoallv` and `Allreduce`.

## Features

- R-MAT graph generation or edge-list input
- Hash and range partitioning
- CSR adjacency storage on each rank
- Distributed BFS, PageRank, and connected components
- Communication time, payload, load-balance, and serial-baseline metrics
- Optional verification against serial implementations

## Requirements

- Python 3.10 or later
- An MPI implementation such as Open MPI or MPICH
- Packages in `requirements.txt`

```powershell
python -m pip install -r requirements.txt
```

## Run

Generate and benchmark an R-MAT graph on four MPI processes:

```powershell
mpiexec -n 4 python dgp.py --scale 18 --partition hash --verify
```

Use an edge-list input file instead:

```powershell
mpiexec -n 4 python dgp.py --input data\sample.edgelist --partition range --verify
```

Each input line contains two non-negative, zero-based vertex IDs, separated by whitespace. Vertex IDs must form a contiguous range from `0` through the largest ID.

Useful options:

```text
--algos bfs,pagerank,cc    Choose algorithms to benchmark
--shuffle                  Randomly relabel generated graph vertices
--no-serial                Skip serial baseline timing
--source 0                 Set the BFS source vertex
```

SciPy is optional. When installed, the program also uses it as an independent connected-components check during `--verify`.

## GUI dashboard

Open `dist\\HPCGraphVisualizer\\HPCGraphVisualizer.exe` to configure a run and view the results. The dashboard shows a results table and bar charts for parallel runtime, communication time, and compute imbalance. It requires Microsoft MPI or another compatible `mpiexec` installation for multi-process benchmarks.

