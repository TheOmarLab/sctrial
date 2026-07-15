# HPC Benchmark Setup Guide

This guide covers setting up and running the sctrial benchmark on a SLURM HPC cluster.

## Overview

The benchmark uses two separate environments:

| Layer | Tool | What it manages |
|-------|------|----------------|
| Python | conda (`sctrial_benchmark`) | Python packages, sctrial source install |
| R | HPC module (`R/4.4.2`) | R and Bioconductor packages in `~/R/library` |

These are kept separate to avoid conda/R library conflicts. R packages are installed once to `~/R/library` and are not part of the conda environment.

---

## Prerequisites

- conda or miniconda installed and on PATH
- Access to the following HPC modules (check with `module avail`):

```
gcc/11.2.0
openblas/dynamic/0.3.18
gsl/2.7.1
R/4.4.2
cmake-3.23.1-gcc-11.2.0-idhlovt
```

> **Different cluster?** Find equivalents with `module avail R`, `module avail cmake`, etc. Update the `module load` lines in `install_r.sh`, `slurm_benchmark.sh`, and `slurm_sensitivity.sh` to match.

---

## One-time Setup

### Step 1 — Clone the repo and check out the correct branch

```bash
git clone <repo-url>
cd sctrial
git checkout <your-branch>
```

### Step 2 — Set up the Python conda environment

Run interactively on the login node (~5–10 min):

```bash
bash scripts/setup_hpc_env.sh
```

This creates the conda environment `sctrial_benchmark` and installs sctrial from source as an editable install. You only need to re-run this if `src/sctrial/benchmark/sctrial_bench_environment.yml` changes.

### Step 3 — Install R packages

Submit as a SLURM job (~1–2h):

```bash
sbatch scripts/install_r.sh
```

Monitor progress:
```bash
tail -f r_install_<jobid>.out
```

Success looks like:
```
OK: edgeR 4.4.2
OK: limma 3.62.2
OK: variancePartition 1.36.3
OK: dreamlet 1.4.1
OK: nebula 1.5.8
All R packages installed and verified.
```

Steps 2 and 3 are independent and can run at the same time.

### Step 4 — Update the working directory path

Both benchmark SLURM scripts contain a placeholder `cd` path. Update it to your repo location:

```bash
# In scripts/slurm_benchmark.sh and scripts/slurm_sensitivity.sh, change:
cd /PATH/TO/sctrial   # ← update to your HPC project path
# to:
cd /your/actual/path/to/sctrial
```

---

## Running the Benchmarks

Both jobs can be submitted at the same time.

### Sensitivity benchmark 

```bash
sbatch scripts/slurm_sensitivity.sh
```

- **Time limit**: 72h
- **Resources**: 1 node, 32 CPUs, 256 GB RAM
- **Scenarios**: 20 parameter combinations × 200 iterations × 4 methods
- **Output**: `manuscript/benchmark/sensitivity/`
- **SLURM logs**: `sensitivity_<jobid>.out` / `sensitivity_<jobid>.err`

### Main simulation benchmark

```bash
sbatch scripts/slurm_benchmark.sh
```

- **Time limit**: 48h
- **Resources**: 1 node, 32 CPUs, 256 GB RAM
- **Output**: `manuscript/benchmark/simulation/`
- **SLURM logs**: `benchmark_<jobid>.out` / `benchmark_<jobid>.err`

---

## Output Files

| Path | Contents |
|------|----------|
| `manuscript/benchmark/simulation/` | Per-scenario CSV results from the main benchmark |
| `manuscript/benchmark/sensitivity/` | Per-scenario CSV results from the sensitivity benchmark |
| `benchmark_<jobid>.out` / `.err` | SLURM stdout/stderr for the main benchmark job |
| `sensitivity_<jobid>.out` / `.err` | SLURM stdout/stderr for the sensitivity benchmark job |
| `r_install_<jobid>.out` / `.err` | SLURM logs for the R install job |

Output CSVs are written incrementally — completed scenarios are saved as they finish, so partial results are preserved if the job times out.

---

## How the Code Is Used

sctrial is installed as an **editable install** (`pip install -e .`). This means Python imports directly from `src/sctrial/` in your repo checkout — **the branch you have checked out is the code that runs**. After pulling changes:

```bash
git pull
# No reinstall needed — editable install picks up changes automatically
sbatch scripts/slurm_sensitivity.sh
```

---

## Updating Environments

### Python packages changed

```bash
bash scripts/setup_hpc_env.sh   # re-runs conda env update + pip install
```

### R packages need updating

```bash
sbatch scripts/install_r.sh
```

---

## Troubleshooting

### Check module names on your cluster

```bash
module avail R 2>&1 | grep -i "^R/"
module avail cmake
module avail gcc
find /apps -name "nlopt.h" 2>/dev/null | head -5   # for nloptr
```

### Verify R packages are installed correctly

```bash
module load gcc/11.2.0 openblas/dynamic/0.3.18 gsl/2.7.1 R/4.4.2
export R_LIBS_USER="$HOME/R/library"
Rscript -e 'library(dreamlet); library(nebula); cat("OK\n")'
```

### R session errors in benchmark output

R errors from dreamlet/nebula appear in the benchmark `.err` file as:
```
dreamlet failed: R error on /tmp/.../run_dreamlet.R: <error message>
```
These are per-iteration failures caught by the benchmark harness. A small number is expected for edge-case simulation scenarios. Systematic failures across all iterations indicate an R package or environment issue.
