"""
Flask server for receiving YCSB benchmark requests from Obsidian.

Expected YAML format from Obsidian:
```yaml
warmup_runs: 2
num_records: 1_000_000
scan_size: 100
distribution: "uniform"
zipfian_theta: 0.99
orderedinserts: true
data_integrity: true
output_file: "AutoGPUDBBenchmarkResult.json"
results_folder: "benchmark/results"
ops: 100
target_qps: 0
min_field_length: 10
max_field_length: 10
field_count: 16

operation_sets:
    Multi_Set_1024:
        multiset: [1, 1024]
    Multi_Set_2048:
        multiset: [1, 2048]

bindings:
  hkv_baseline:
    binding_type: "cpp"
    gpu_ids: [1]
    value_type: "float"
    hbm_gb: 15
    gpu_init_capacity: 500_000
    gpu_max_capacity: 10_000_000
    multiset_operation: "accum_or_assign"

  hkv_kernelopt:
    binding_type: "cpp"
    gpu_ids: [1]
    value_type: "float"
    hbm_gb: 15
    gpu_init_capacity: 500_000
    gpu_max_capacity: 10_000_000
    multiset_operation: "accum_or_assign"
```

Batch run with variables:
- POST body can include "variables": ["operation_sets", "num_records", ...]
- If "operation_sets" is in variables, iterate over all operation_sets defined in YAML
- For other variables, if the value is a list, iterate over each value
- Multiple variables create a cartesian product of all combinations
```
"""

import os
import sys
import yaml
import traceback
import threading
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

from gycsb.Runner import generate_variable_combinations, run_all_benchmarks, run_single_benchmark


app = Flask(__name__)
CORS(app)  # Enable CORS for Obsidian requests

# Global lock to ensure only one benchmark runs at a time
_benchmark_lock = threading.Lock()
_benchmark_running = False
_current_benchmark_name = None


def parse_benchmark_config(yaml_content: str) -> dict:
    """Parse YAML content and extract workload and bindings configurations.
    
    Supports format:
    - Top-level workload config (num_records, ops, operations, etc.)
    - Multiple bindings under 'bindings' key
    """
    # Convert tabs to spaces (common issue with copy-pasted YAML)
    yaml_content = yaml_content.replace('\t', '  ')
    
    config = yaml.safe_load(yaml_content)
    
    if 'bindings' not in config:
        raise ValueError("Invalid YAML format: must contain 'bindings' section")
    
    return config



@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'YCSB Benchmark Server'
    })


@app.route('/status', methods=['GET'])
def benchmark_status():
    """Check if a benchmark is currently running."""
    with _benchmark_lock:
        return jsonify({
            'benchmark_running': _benchmark_running,
            'current_benchmark': _current_benchmark_name,
            'timestamp': datetime.now().isoformat()
        })


def run_single_benchmark_task(config: dict, running_name: str):
    """
    Background task to run a single benchmark for all bindings.
    """
    global _benchmark_running, _current_benchmark_name
    try:
        results = run_single_benchmark(config, running_name)
        return results
    except Exception as e:
        print(f"[Background Task Error] {str(e)}")
        traceback.print_exc()
    finally:
        # Always reset the running state when task finishes
        with _benchmark_lock:
            _benchmark_running = False
            _current_benchmark_name = None
            print(f"[Background Task] Released benchmark lock")


@app.route('/benchmark', methods=['POST'])
def run_benchmark_endpoint():
    """
    Run a single YCSB benchmark with the provided YAML configuration.
    Validates the request and returns 200 immediately, running benchmark in background.
    
    Expected JSON body:
    {
        "filePath": "path/to/file.yaml",
        "yamlContent": "num_records: 1000000\\noperations:\\n  multiget: [1, 65536]\\nbindings:\\n  ...",
        "running_name": "my_benchmark"
    }
    """
    try:
        # Parse request body
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON body provided'
            }), 400
        
        file_path = data.get('filePath', None)
        yaml_content = data.get('yamlContent')
        running_name = data.get('running_name', None)
        
        if not yaml_content:
            return jsonify({
                'success': False,
                'error': 'No yamlContent provided'
            }), 400
            
        if not running_name:
            return jsonify({
                'success': False,
                'error': 'No running_name provided'
            }), 400
        
        print(f"\n{'#'*60}")
        print(f"[Single Benchmark Request] Received request for: {file_path}")
        print(f"{'#'*60}")
        print(f"[Single Benchmark Request] YAML content:\n{yaml_content}")
        
        # Parse and validate YAML configuration
        config = parse_benchmark_config(yaml_content)
        bindings = config['bindings']
        
        # process the operations
        operation_sets = config.get('operations', None)
        if operation_sets is None:
            raise ValueError("operations is required")
        operations = next(iter(operation_sets.values()))
        config['operations'] = operations
        
        binding_names = list(bindings.keys())
        if not binding_names:
            raise ValueError("No bindings found in configuration")
        
        # Check if a benchmark is already running
        global _benchmark_running, _current_benchmark_name
        with _benchmark_lock:
            if _benchmark_running:
                return jsonify({
                    'success': False,
                    'error': f'A benchmark is already running: {_current_benchmark_name}',
                    'current_benchmark': _current_benchmark_name
                }), 409  # 409 Conflict
            
            # Set running state before starting
            _benchmark_running = True
            _current_benchmark_name = running_name
        
        # Validation passed, start background task
        thread = threading.Thread(
            target=run_single_benchmark_task,
            args=(config, running_name),
            daemon=True
        )
        thread.start()
        
        print(f"[Single Benchmark] Started background task for {running_name}")
        
        # Return immediately with 200
        return jsonify({
            'success': True,
            'message': f'Benchmark started for {file_path}',
            'filePath': file_path,
            'bindings': binding_names,
            'running_name': running_name,
            'results_file': f'{running_name}.json',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except yaml.YAMLError as e:
        error_msg = f'YAML parsing error: {str(e)}'
        print(f"[Error] {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 400
        
    except ValueError as e:
        error_msg = f'Configuration error: {str(e)}'
        print(f"[Error] {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 400
        
    except Exception as e:
        error_msg = f'Request validation error: {str(e)}'
        print(f"[Error] {error_msg}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback.format_exc()
        }), 500


def run_batch_benchmark_task(config: dict, running_name: str, variables: list = None):
    """
    Background task to run batch benchmarks with variable combinations.
    """
    global _benchmark_running, _current_benchmark_name
    try:
        run_all_benchmarks(config, running_name, variables=variables)
        print(f"\n{'='*60}")
        print(f"[Background Task] Batch benchmark completed for {running_name}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"[Background Task Error] {str(e)}")
        traceback.print_exc()
    finally:
        # Always reset the running state when task finishes
        with _benchmark_lock:
            _benchmark_running = False
            _current_benchmark_name = None
            print(f"[Background Task] Released benchmark lock")


@app.route('/benchmarks', methods=['POST'])
def run_benchmarks_endpoint():
    """
    Run YCSB benchmark with the provided YAML configuration.
    Validates the request and returns 200 immediately, running benchmark in background.
    
    Expected JSON body:
    {
        "filePath": "path/to/file.yaml",
        "yamlContent": "num_records: 1000000\\nbindings:\\n  hkv_baseline:\\n    ...",
        "variables": ["operation_sets", "num_records", ...],  // Optional: for batch runs
        "running_name": "my_benchmark"
    }
    
    Variables behavior:
    - If "operation_sets" is in variables, iterate over all operation_sets defined in YAML
    - For other variables, if the value in YAML is a list, iterate over each value
    - Multiple variables create a cartesian product of all combinations
    """
    try:
        # Parse request body
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON body provided'
            }), 400
        
        file_path = data.get('filePath', 'unknown')
        yaml_content = data.get('yamlContent')
        variables = data.get('variables')  # Optional list of variables for batch run
        running_name = data.get('running_name', None)
        
        if not yaml_content:
            return jsonify({
                'success': False,
                'error': 'No yamlContent provided'
            }), 400
            
        if not running_name:
            return jsonify({
                'success': False,
                'error': 'No running_name provided'
            }), 400
        
        print(f"\n{'#'*60}")
        print(f"[Benchmark Request] Received request for: {file_path}")
        print(f"[Benchmark Request] Variables for batch run: {variables}")
        print(f"{'#'*60}")
        
        # Parse and validate YAML configuration
        config = parse_benchmark_config(yaml_content)
        
        bindings = config['bindings']
        binding_names = list(bindings.keys())
        if not binding_names:
            raise ValueError("No bindings found in configuration")
        
        # Calculate expected runs for response
        combinations = generate_variable_combinations(config, variables or [])
        total_runs = len(combinations) * len(binding_names)
        
        # Check if a benchmark is already running
        global _benchmark_running, _current_benchmark_name
        with _benchmark_lock:
            if _benchmark_running:
                return jsonify({
                    'success': False,
                    'error': f'A benchmark is already running: {_current_benchmark_name}',
                    'current_benchmark': _current_benchmark_name
                }), 409  # 409 Conflict
            
            # Set running state before starting
            _benchmark_running = True
            _current_benchmark_name = running_name
        
        # Validation passed, start background task
        thread = threading.Thread(
            target=run_batch_benchmark_task,
            args=(config, running_name, variables),
            daemon=True
        )
        thread.start()
        
        print(f"[Batch Benchmark] Started background task for {running_name}")
        
        # Return immediately with 200
        return jsonify({
            'success': True,
            'message': f'Batch benchmark started for {file_path}',
            'filePath': file_path,
            'bindings': binding_names,
            'variables': variables,
            'combinations_count': len(combinations),
            'total_runs': total_runs,
            'running_name': running_name,
            'results_file': f'{running_name}.json',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except yaml.YAMLError as e:
        error_msg = f'YAML parsing error: {str(e)}'
        print(f"[Error] {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 400
        
    except ValueError as e:
        error_msg = f'Configuration error: {str(e)}'
        print(f"[Error] {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 400
        
    except Exception as e:
        error_msg = f'Request validation error: {str(e)}'
        print(f"[Error] {error_msg}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback.format_exc()
        }), 500


def create_app():
    """Factory function to create the Flask app."""
    return app