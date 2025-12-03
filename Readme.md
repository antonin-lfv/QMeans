<h1 align="center">
  <img src="https://github.com/antonin-lfv/QMeans/assets/63207451/79c11b83-111e-4f70-baf2-6b4246789d3d" width="260">
<br>
</br>
  QMeans Algorithm
</h1>

<h4 align="center">A quantum version of the KMeans algorithm, using Qiskit</h4>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
  <a href="https://doi.org/10.1016/B978-0-443-29096-1.00005-2">
    <img src="https://img.shields.io/badge/DOI-10.1016-orange.svg" alt="DOI">
  </a>
  <!-- Badge pour la version fr de l'article -->
  <a href="https://antonin-lfv.github.io/assets/pdf/QMeans.pdf">
    <img src="https://img.shields.io/badge/Version-française-blue.svg" alt="Version française">
  </a>
</p>

<br>

## ⚛️ About The Project

While the classical K-Means algorithm is a staple in unsupervised learning, it suffers from time complexity limitations of $O(iknm)$ (where $i$ is iterations, $k$ clusters, $n$ points, and $m$ dimensions). **QMeans** explores how quantum computing can address these limitations.

This project implements a **hybrid sequential approach**. Unlike the purely theoretical proposal by *Kerenidis et al.*, which relies on a single deep quantum circuit (currently unfeasible), this implementation breaks down the process into sequential quantum subroutines managed by a classical controller. This design allows the algorithm to run on current NISQ (Noisy Intermediate-Scale Quantum) devices and simulators.

> **Fun Fact:** Validating this project required executing over **100,000 quantum circuits** on IBM's servers.

## ⚙️ How It Works (The Hybrid Loop)

The algorithm follows a specific loop where distance estimation and cluster assignment are offloaded to the quantum processor (QPU), while centroid updates remain classical.

### 0. Initialization (Quantum K-Means++)
Centroids are initialized using a quantum variation of the K-Means++ algorithm to optimize convergence speed.

### 1. Quantum Distance Estimation
We use the **Swap Test** logic to estimate the Euclidean distance between a data point $x$ and a centroid $y$ via their dot product. The probability of measuring the auxiliary qubit in state $|1\rangle$ is related to the overlap:

$$P(1_{aux}) = \frac{1}{2} - \frac{1}{2} \langle x | y \rangle^2$$

*Hardware Requirement:* $3n$ qubits are required for distance estimation on $n$ clusters.

### 2. Quantum Cluster Assignment
Once distances are estimated, we identify the nearest centroid using the **Quantum Bit String Comparator** associated with the **Dürr-Hoyer algorithm** (a quantum minimum finding algorithm).

*Hardware Requirement:* $5n$ qubits are required to find the minimum in a list of integers on $n$ bits.

### 3. Classical Centroid Update
The new centroids are calculated classically by computing the barycenter of the assigned points for each cluster $C_i$:

$$c_i = \frac{1}{|C_i|} \sum_{x \in C_i} x$$

## Getting Started

Start by creating the file `config.py` in the root directory with the following content (replace the placeholders with your actual API tokens):

```python
IBM_QUANTUM_API_TOKEN = "YOUR_IBM_QUANTUM_API_TOKEN"
IONQ_API_TOKEN = "YOUR_IONQ_API_TOKEN"
```

Then, install the environment using UV:

```bash
uv sync
```

And run the main script:

```bash
uv run main.py
```

## ❝ Citation

If you use this code or the paper in your research, please cite the following:

```bib
@incollection{LEFEVRE2025115,
title = {Chapter 7 - Hybrid quantum-classical framework for clustering},
editor = {Rajkumar Buyya and Sukhpal Singh Gill},
booktitle = {Quantum Computing},
publisher = {Morgan Kaufmann},
pages = {115-137},
year = {2025},
isbn = {978-0-443-29096-1},
doi = {https://doi.org/10.1016/B978-0-443-29096-1.00005-2},
url = {https://www.sciencedirect.com/science/article/pii/B9780443290961000052},
author = {Antonin Lefevre},
} 
```
