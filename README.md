# gYCSB

<div align="center">
  <img src="logo.png" alt="gYCSB Logo" width="300">
</div>

A general-purpose YCSB (Yahoo! Cloud Serving Benchmark) implementation supporting CPU, CUDA, and distributed execution with Python and C++ bindings.

[YCSB](https://github.com/brianfrankcooper/YCSB) is an industry-standard benchmark framework for evaluating the performance of key-value and cloud serving stores. This implementation extends YCSB with GPU acceleration and distributed benchmarking capabilities.

## Features

- **Multi-platform Support**: CPU and CUDA (GPU) execution backends
- **Distributed Benchmarking**: Run benchmarks across multiple nodes
- **Python & C++ APIs**: Flexible programming interfaces for both languages
- **Interactive Visualizer**: Web-based dashboard for analyzing benchmark results
- **Workload Management**: RESTful API server for submitting and managing benchmark tasks
- **Obsidian Integration**: Plugin support for seamless workflow integration

## Prerequisites

- Python 3.12+
- CUDA 12.8+
- CMake 3.30.4+
- GCC 11+ (Linux)
- Conda (for environment management)

## Installation

### 1. Create Conda Environment

```bash
conda env create -f env.yml
conda activate gYCSB
```

### 2. Install Python Package

```bash
pip install -e .
```

### 3. Build C++/CUDA Components

```bash
mkdir -p build && cd build
cmake .. -Dsm=86 && make -j${nproc} && cmake --install .
```

**Note**: Adjust the `-Dsm=86` flag to match your GPU's compute capability (e.g., `-Dsm=75` for Turing, `-Dsm=89` for Ada Lovelace).

## Usage

### Running Benchmarks

Example scripts are provided in the `example/` directory:

- **Single Node**: `example/single_run.py` - Run benchmarks on a single machine
- **Batch Execution**: `example/batch_run.py` - Run multiple benchmark configurations
- **Distributed Batch**: `example/batch_run_dims.py` - Batch execution for distributed setups

For distributed benchmarking across multiple nodes, see `run_distributed_benchmark.py` in the root directory.

### Visualizer

Launch the interactive web-based visualizer to analyze benchmark results:

```bash
python -m visualizer
```

The visualizer provides real-time performance metrics, throughput analysis, and latency distributions.

### Workload Manager (Obsidian Server)

Start the RESTful API server for managing benchmark tasks:

```bash
python -m obsidian_server
```

Once running, submit benchmark tasks via the API endpoint:
- **API Base URL**: `http://localhost:5000/benchmarks`

For Obsidian note-taking integration, install the [obsidian-gycsb-plugin](https://github.com/haiqiang-zhang/obsidian-gycsb-plugin) first.

## License

MIT License - see [LICENSE](LICENSE) file for details.