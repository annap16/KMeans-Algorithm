# Choose your language version
- [Polski](#kmeans-pol)
- [English](#kmeans-en)

--- 
# Projekt  <a name="kmeans-pol"></a>
To repozytorium zawiera implementację algorytmu klastrowania K-średnich w trzech różnych wariantach:
- **K-średnich na CPU**: Standardowa implementacja działająca na procesorze.
- **K-średnich na GPU (oparte na kernelach)**: Wykorzystuje kernele CUDA do równoległych obliczeń.
- **K-średnich na GPU (oparte na Thrust)**: Wykorzystuje bibliotekę Thrust firmy NVIDIA do wysokopoziomowej równoległości.

# Format Uruchamiania Programu

Przed uruchomieniem programu należy go skompilować za pomocą:

```
make
```

Następnie można go uruchomić z linii poleceń w następujący sposób:

```
KMeans data_format computation_method input_file output_file
```

- **data_format**: `txt` (tekstowy) lub `bin` (binarny)
- **computation_method**:
  - `cpu` → Standardowa implementacja CPU
  - `gpu1` → W pełni oparta na GPU metoda z optymalizacjami pamięci współdzielonej
  - `gpu2` → Hybrydowe podejście GPU wykorzystujące zarówno kernele, jak i operacje Thrust
- **input_file**: Ścieżka do pliku z danymi (`txt` lub `bin`)
- **output_file**: Ścieżka do pliku wynikowego

# Opis Algorytmu K-średnich
K-średnich to algorytm klastrowania, który dzieli zbiór danych na K klastrów. Jest to algorytm iteracyjny, minimalizujący wariancję wewnątrz każdego klastra.

Etapy algorytmu:
1. Inicjalizacja K centroidów losowo.
2. Przypisanie każdego punktu danych do najbliższego centroidu.
3. Aktualizacja centroidów poprzez obliczenie średniej współrzędnych przypisanych punktów.
4. Powtarzanie kroków 2 i 3 do momentu, gdy centroidy przestaną się zmieniać (konwergencja).

# Główne Założenia dla Każdej Implementacji

## Algorytm 1 (oparty na kernelach):
Implementacja ta wykorzystuje w pełni kernele GPU z optymalizacjami pamięci współdzielonej. Każdy wątek oblicza odległości i przypisuje punkty do klastrów, jednocześnie zarządzając lokalną akumulacją sum centroidów i liczności klastrów w pamięci współdzielonej. Synchronizacja między blokami zapewnia poprawną aktualizację centroidów.

## Algorytm 2 (oparty na Thrust):
Ta implementacja łączy jawne wywołania kernelów z operacjami Thrust dla optymalizacji wydajności. Przypisanie punktów do klastrów odbywa się w kernelu GPU, natomiast aktualizacja centroidów wykorzystuje sortowanie i redukcję w Thrust, co pozwala na efektywne agregowanie danych.

## Bazowa implementacja CPU:
Standardowy algorytm K-średnich wykonywany w całości na CPU. Każda iteracja oblicza odległości, przypisuje punkty do klastrów i aktualizuje centroidy sekwencyjnie. Służy jako punkt odniesienia do porównania przyspieszenia uzyskanego na GPU.

# Charakterystyka Danych
- **Liczba punktów (N)**: Do 50 milionów
- **Liczba wymiarów (d)**: Od 1 do 20
- **Liczba klastrów (k)**: Od 2 do 20
- **Inicjalne centroidy**: Pierwsze k punktów ze zbioru danych
- **Kryteria stopu**: Brak zmian w klastrach lub osiągnięcie limitu 100 iteracji
- **Obsługiwane formaty wejściowe**:
  - `txt`
  - `bin`
- **Format wyników**:
  - Pierwsze k linii: Współrzędne centroidów
  - Kolejne N linii: ID klastra dla każdego punktu

---

# Project  <a name="kmeans-en"></a>
This repository contains an implementation of the K-Means clustering algorithm with three different solutions:
- **CPU-based K-Means:** A standard implementation running on the CPU.
- **GPU-based K-Means (Kernel-based):** Uses CUDA kernels for parallel computation.
- **GPU-based K-Means (Thrust-based):** Uses NVIDIA's Thrust library for high-level parallelization.

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
- **Number of points (N):** Up to 50 million
- **Dimensionality (d):** Between 1 and 20
- **Number of clusters (k):** Between 2 and 20
- **Initial centroids:** First k points from the dataset
- **Stopping criteria:** No cluster changes or a maximum of 100 iterations
- **Accepted input data format:**
  - `txt`
  - `bin`
- **Result Format:**
  - First k lines: Centroid coordinates
  - Next N lines: Cluster ID for each point

