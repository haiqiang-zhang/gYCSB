from benchmark.ycsb.Runner import run_single_benchmark
from benchmark.ycsb.ConfigLoader import __load_config
import os
from pathlib import Path

def main():
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    path = Path(this_file_path) 
    
    config = __load_config(path / "new_ycsb_config_format.yaml")
    run_single_benchmark(config, "test_single_run")


if __name__ == "__main__":
    main()