from .YCSBController import YCSBController
import itertools
import os
import json

RESULTS_FOLDER = "/pub/nfs-data/zhaiqiang/autogpudb/benchmark/results"


def run(workload_config: dict, binding_name: str, binding_config: dict, warmup: bool = False) -> dict:
    """Run a single YCSB benchmark with the given configurations."""
    binding_type = binding_config.get('binding_type', 'cpp')
    gpu_ids = binding_config.get('gpu_ids', [0])
    value_type = binding_config.get('value_type', 'float')
    
    # Extract workload parameters with defaults
    num_records = workload_config.get('num_records', 1_000_000)
    operations = workload_config.get('operations', None)
    workload_name = workload_config.get('workload_name', None)
    distribution = workload_config.get('distribution', 'uniform')
    zipfian_theta = workload_config.get('zipfian_theta', 0.99)
    orderedinserts = workload_config.get('orderedinserts', True)
    data_integrity = workload_config.get('data_integrity', True)
    min_field_length = workload_config.get('min_field_length', 10)
    max_field_length = workload_config.get('max_field_length', 10)
    field_count = workload_config.get('field_count', 16)
    key_type = workload_config.get('key_type', 'np_int64')
    num_ops = workload_config.get('ops', 100)
    generator_num_processes = workload_config.get('generator_num_processes', 50)
    
    controller = YCSBController(
        num_records=num_records,
        operations=operations,
        workload_name=workload_name,
        distribution=distribution,
        zipfian_theta=zipfian_theta,
        orderedinserts=orderedinserts,
        data_integrity=data_integrity,
        min_field_length=min_field_length,
        max_field_length=max_field_length,
        field_count=field_count,
        binding_type=binding_type,
        binding_name=binding_name,
        binding_config=binding_config,
        gpu_device=gpu_ids,
        generator_num_processes=generator_num_processes,
        value_type=value_type,
        key_type=key_type
    )
    
    results = controller.run(num_ops=num_ops, num_streams=0, warmup=warmup)
    return results


def generate_variable_combinations(config: dict, variables: list) -> list:
    """
    Generate all combinations of variable values for batch running.
    
    Args:
        config: The parsed YAML config
        variables: List of variable names to iterate over
        
    Returns:
        List of dicts, each containing one combination of variable values
        Each dict has:
        - For operation_sets: {'operation_set_name': str, 'operations': dict}
        - For other variables: {variable_name: value}
    """
    if not variables:
        return [{}]
    
    variable_values = {}
    
    for var_name in variables:
        if var_name == 'operation_sets':
            # Special handling for operation_sets
            operation_sets = config.get('operation_sets', {})
            if not operation_sets:
                print(f"[Warning] 'operation_sets' requested but not found in config")
                continue
            # Each operation_set is a dict like {'Multi_Set_1024': {'multiset': [1, 1024]}}
            # We store tuples of (name, operations_dict)
            variable_values['operation_sets'] = [
                {'workload_name': name, 'operations': ops}
                for name, ops in operation_sets.items()
            ]
        else:
            # For other variables, check if the value is a list
            value = config.get(var_name)
            if value is None:
                print(f"[Warning] Variable '{var_name}' not found in config")
                continue
            if isinstance(value, list):
                # If it's a list, iterate over each value
                variable_values[var_name] = [{var_name: v} for v in value]
            else:
                # If not a list, use as single value
                variable_values[var_name] = [{var_name: value}]
    
    if not variable_values:
        return [{}]
    
    # Generate cartesian product of all variable values
    keys = list(variable_values.keys())
    value_lists = [variable_values[k] for k in keys]
    
    combinations = []
    for combo in itertools.product(*value_lists):
        merged = {}
        for item in combo:
            merged.update(item)
        combinations.append(merged)
    
    return combinations



def apply_variable_combination(workload_config: dict, combination: dict) -> dict:
    """
    Apply a variable combination to the workload config.
    
    Args:
        workload_config: The base workload config
        combination: A dict of variable values to apply
        
    Returns:
        Modified workload config with the combination applied
    """
    modified_config = workload_config.copy()
    if combination == {}:
        modified_config['operations'] = next(iter(workload_config['operation_sets'].values()))
        modified_config['workload_name'] = next(iter(workload_config['operation_sets'].keys()))
        return modified_config
        
    for key, value in combination.items():
        if key == 'workload_name':
            # This is metadata, store it but don't override workload params
            modified_config['workload_name'] = value
        elif key == 'operations':
            # This comes from operation_sets, override the operations
            modified_config['operations'] = value
        else:
            # Regular variable, override directly
            modified_config[key] = value
    
    return modified_config


def run_all_benchmarks(config: dict, running_name: str, variables: list = None) -> dict:
    """
    Run benchmarks for all bindings in the config.
    Similar to single_run.py: warmup with first binding, then run all bindings.
    
    Args:
        config: The parsed YAML config
        variables: Optional list of variables to iterate over for batch running
    """
    bindings = config['bindings']
    warmup_runs = config.get('warmup_runs', 2)
    
    all_results = {}
    binding_names = list(bindings.keys())
    
    if not binding_names:
        raise ValueError("No bindings found in configuration")
    
    # Generate variable combinations
    combinations = generate_variable_combinations(config, variables or [])
    total_runs = len(combinations) * len(binding_names)
    
    print(f"\n{'='*60}")
    print(f"Batch Run Configuration:")
    print(f"  - Variables: {variables or 'None'}")
    print(f"  - Combinations: {len(combinations)}")
    print(f"  - Bindings: {len(binding_names)}")
    print(f"  - Total runs: {total_runs}")
    print(f"{'='*60}\n")
    
    # Warmup with first binding and first combination
    first_binding_name = binding_names[0]
    first_binding_config = bindings[first_binding_name]
    first_workload_config = apply_variable_combination(config, combinations[0] if combinations else {})
    print(f"\n{'='*60}")
    print(f"Starting warmup phase with binding: {first_binding_name}")
    print(f"{'='*60}\n")
    
    for i in range(warmup_runs):
        print(f"[Warmup {i+1}/{warmup_runs}]")
        run(first_workload_config, first_binding_name, first_binding_config, warmup=True)
    
    # Run actual benchmarks for all combinations and bindings
    
    run_idx = 0
    for combo_idx, combination in enumerate(combinations):
        combo_name = combination.get('workload_name', f'combo_{combo_idx}')
        workload_config = apply_variable_combination(config, combination)
        
        print(f"\n{'='*60}")
        print(f"[Combination {combo_idx+1}/{len(combinations)}] {combo_name}")
        print(f"  Variables: {combination}")
        print(f"{'='*60}")
        
        for binding_name in binding_names:
            run_idx += 1
            binding_config = bindings[binding_name]
            
            print(f"\n[Run {run_idx}/{total_runs}] {combo_name} + {binding_name}")
            print("-" * 40)
            
            result = run(workload_config, binding_name, binding_config, warmup=False)
            
            # Store results with combination info
            result_key = f"{combo_name}:{binding_name}"
            all_results[result_key] = {
                'combination': combination,
                'binding_name': binding_name,
                'results': result
            }
            
            results = []
            if not os.path.exists(f"{RESULTS_FOLDER}/{running_name}.json"):
                with open(f"{RESULTS_FOLDER}/{running_name}.json", "w") as f:
                    json.dump([], f, indent=4)
            else:
                with open(f"{RESULTS_FOLDER}/{running_name}.json", "r") as f:
                    results = json.load(f)
            results.append(result)
            with open(f"{RESULTS_FOLDER}/{running_name}.json", "w") as f:
                json.dump(results, f, indent=4)
            print(f"Workload {running_name} results have been written to {RESULTS_FOLDER}/{running_name}.json")

    return all_results




def run_single_benchmark(config: dict, running_name: str):
    """
    Background task to run a single benchmark for all bindings.
    """
    bindings = config['bindings']
    warmup_runs = config.get('warmup_runs', 2)
    binding_names = list(bindings.keys())
    
    
    # modify the operation_sets to operations
    operation_sets = config.get('operation_sets', {})
    if operation_sets is None:
        raise ValueError("operation_sets is required")
    config['operations'] = next(iter(operation_sets.values()))
    config['workload_name'] = next(iter(operation_sets.keys()))
    
    
    # Warmup with first binding
    first_binding_name = binding_names[0]
    first_binding_config = bindings[first_binding_name]
    
    for i in range(warmup_runs):
        print(f"[Warmup {i+1}/{warmup_runs}]")
        run(config, first_binding_name, first_binding_config, warmup=True)
    
    # Run actual benchmarks for all bindings
    
    for binding_name in binding_names:
        binding_config = bindings[binding_name]
        
        result = run(config, binding_name, binding_config, warmup=False)
        
        # Append to the results file
        results = []
        if not os.path.exists(f"{RESULTS_FOLDER}/{running_name}.json"):
            with open(f"{RESULTS_FOLDER}/{running_name}.json", "w") as f:
                json.dump([], f, indent=4)
        else:
            with open(f"{RESULTS_FOLDER}/{running_name}.json", "r") as f:
                results = json.load(f)
        results.append(result)
        with open(f"{RESULTS_FOLDER}/{running_name}.json", "w") as f:
            json.dump(results, f, indent=4)
        print(f"Workload {running_name} results have been written to {RESULTS_FOLDER}/{running_name}.json")

    return results