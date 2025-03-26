# Project
This repository contains an implementation of the K-Means clustering algorithm with three different solutions:
- CPU-based K-Means: A standard implementation running on the CPU.
- GPU-based K-Means (Kernel-based): Uses CUDA kernels for parallel computation.
- GPU-based K-Means (Thrust-based): Uses NVIDIA's Thrust library for high-level parallelization.

# Program Execution Format

Before running the program, compile it using:

```
make
```

Then, execute it from the command line as:

```
KMeans data_format computation_method input_file output_file
```

- **data_format**: Either `txt` (text) or `bin` (binary)
- **computation_method**:
  - `cpu` → Standard CPU-based computation
  - `gpu1` → Fully GPU-based kernel approach with shared memory optimizations
  - `gpu2` → Hybrid GPU approach with explicit kernel calls & Thrust operations
- **input_file**: Path to the dataset file (`txt` or `bin`)
- **output_file**: Path to save results

# K-Means Algorithm - Description
K-Means is a clustering algorithm that partitions a dataset into K clusters. It is an iterative algorithm that minimizes the variance within each cluster.

Algorithm Steps:
1. Initialize K cluster centroids randomly.
2. Assign each data point to the nearest centroid.
3. Update centroids by computing the mean of assigned points.
4. Repeat steps 2 and 3 until centroids stop changing (convergence).

# Main Assumptions for Each Implementation

## Algorithm 1 (Kernel-based):
This implementation relies entirely on GPU kernels with shared memory optimizations. Each thread computes distances and updates cluster assignments while maintaining local accumulations of centroid sums and cluster sizes in shared memory. Synchronization between blocks ensures correct centroid updates.


## Algorithm 2 (Thrust-based):
This implementation combines explicit kernel execution with Thrust library operations for performance optimization. Cluster assignments are computed in a GPU kernel, while centroid updates leverage sorting and reduction using Thrust to efficiently aggregate data.


## CPU Baseline Implementation:
A standard K-Means algorithm executed purely on CPU. Each iteration computes distances, assigns clusters, and updates centroids sequentially. This serves as a reference point for comparing GPU-based speedup.

# Data Characteristics
- Number of points (N): Up to 50 million
- Dimensionality (d): Between 1 and 20
- Number of clusters (k): Between 2 and 20
- Initial centroids: First k points from the dataset
- Stopping criteria: No cluster changes or a maximum of 100 iterations
- Accepted input data format:
  - txt
  - bin
- Result Format:
  - First k lines: Centroid coordinates
  - Next N lines: Cluster ID for each point

