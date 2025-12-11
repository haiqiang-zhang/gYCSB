# gYCSB

## Setup
```bash
conda env create -f env.yml
conda activate gYCSB
pip install -e .
mkdir -p build && cd build
cmake .. -Dsm=86 && make -j${nproc} && cmake --install .
```