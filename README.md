# CUDA-Based Integral Image Computation

## Table of Contents
- [Overview](#overview)
  - [Integral Image](#integral-image)
    - [Formula](#formula)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
    - [C++](#c)
    - [Python](#python)
- [Cuda Kernels](#cuda-kernels)
    - [Included Kernels](#included-kernels)
- [Performance](#performance)
- [License](#license)
- [Author](#author)

## Overview
This project implements integral image computation for grayscale images using CUDA. It leverages GPU parallel processing to achieve high performance.

The parallel computation on the GPU is based on the alternation of two kernels in the following order:
1. Row-wise Scan
2. Transpose
3. Row-wise Scan
4. Transpose

Specifically, for the scan kernel, two versions are provided: a naive implementation and an optimized one.

### Integral Image
An integral image, also known as a summed-area table, is a representation that allows for fast computation of the sum of values in a rectangular subset of an image.

For example:

![Integral Image Example](https://i.ibb.co/4wT6rKMg/240px-Integral-image-application-example-svg.png)

#### Formula
Given an input image $i(x, y)$, the integral image $I(x, y)$ is computed as:

$$
I(x,y) = i(x,y) + I(x-1,y)+I(x,y-1)-I(x-1,y-1)
$$

## Repository Structure

```plaintext
.
├── python/
│   ├── pycuda_test.py   # Python script using pyCUDA for invoking CUDA kernels and managing the workflow
│   └── numba_test.py    # Python script using Numba for invoking CUDA kernels and managing the workflow
├── c++/
│   ├── main.cu          # CUDA source file containing benchmarking logic
│   └── kernel.cu        # CUDA kernel definitions
└── integralimage        # Script for compiling the project and running benchmarks
```

## Prerequisites

- CUDA-capable NVIDIA GPU
- CUDA Toolkit
- C++ compiler
- CMake
- Python 3.x (for Python interface)
- Python Libraries:
    - numpy
    - pycuda
    - numba

## Installation
1. Clone the repository:

```sh
git clone https://github.com/AlessioBugetti/integral-image-processing.git
cd integral-image-processing
```
2. Install Python dependencies:

```sh
pip install -r python/requirements.txt
```
3. Ensure the CUDA environment is set up:
    - Install NVIDIA drivers.
    - Install the CUDA Toolkit.
    - Verify with ```nvcc --version```.

## Usage

### C++
```sh
./integralimage build
./integralimage run
```

### Python
```sh
python pycuda_test.py
```
or
```sh
python numba_test.py
```

## Cuda Kernels

### Included Kernels:
- `SumRows`: Naively computes the row-wise scan (prefix sum) of a matrix
- `SinglePassRowWiseScan`: Optimized computation of the row-wise scan (prefix sum) of a matrix
- `Transpose`: Transposes a matrix using block-level tiling with shared memory

## Performance
The implementation includes benchmarking capabilities that measure:
- Sequential CPU execution time
- CUDA execution time for the naive implementation of the integral image computation
- CUDA execution time for the optimized implementation of the integral image computation
- Speedup ratios compared to the CPU implementation for both the naive and optimized implementations
- Measurements are averaged over multiple iterations to ensure reliable results.

## License
This project is licensed under the GPL-3.0-only License. See the [`LICENSE`](LICENSE) file for more details.

## Author
Alessio Bugetti - alessiobugetti98@gmail.com
