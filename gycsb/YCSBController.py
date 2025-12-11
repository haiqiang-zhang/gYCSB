from enum import Enum
import cupy as cp
import random
import json
import os
from typing import Type
import numpy as np
import pandas as pd
from functools import partial
from bindings_py.BindingBase import BindingBase
from .binding_registry import get_binding_class
from libgycsb.python import YCSBBridge, CPPOperation
from .WorkloadGenerator import WorkloadGenerator, _generate_operation_worker, OperationFactory
from concurrent.futures import ThreadPoolExecutor, as_completed as as_completed_thread
from concurrent.futures import ProcessPoolExecutor, as_completed as as_completed_process
import multiprocessing as mp

class BindingType(Enum):
    PYTHON = "python"
    CPP = "cpp"


class YCSBController:
    def __init__(self, num_records: int, 
                 operations=[('read', 0.5), ('update', 0.5)], 
                 workload_name: str = "",
                 distribution: str = "zipfian", 
                 zipfian_theta: float = 0.99,
                 scan_size: int = 100, 
                 orderedinserts: bool = False,
                 data_integrity: bool = True,
                 target_qps: int | None = None,
                 min_field_length: int = 128,
                 max_field_length: int = 512,
                 field_count: int = 5,
                 binding_type: str = "python",
                 value_type: str = "double",
                 binding_name: str = "hkv_baseline",
                 output_file: str | None = None,
                 load_data_output_file: str | None = None,
                 operations_file: str | None = None,
                 gpu_device: list[int] = [0],
                 binding_config: dict = {},
                 generator_num_processes: int = 8,
                 key_type: str = "np_int64"):
        self.num_records = num_records
        self.operations = operations
        self.workload_name = workload_name
        self.scan_size = scan_size
        self.distribution = distribution
        self.zipfian_theta = zipfian_theta
        self.output_file = output_file
        self.load_data_output_file = load_data_output_file
        self.operations_file = operations_file
        self.orderedinserts = orderedinserts
        self.data_integrity = data_integrity
        self.min_field_length = min_field_length
        self.max_field_length = max_field_length
        self.binding_type = BindingType(binding_type)
        self.value_type = value_type
        self.binding_name = binding_name
        self.binding_config = binding_config
        self.field_count = field_count
        self.generator_num_processes = generator_num_processes
        self.key_type = key_type
        
        
        if target_qps is not None and target_qps < 0:
            raise ValueError("Target QPS must be greater than 0 or 0 for no QPS control")
        elif target_qps == 0:
            self.target_qps = None
        else:
            self.target_qps = target_qps
            
        self.workload_gen = WorkloadGenerator(data_integrity=self.data_integrity, 
                                              distribution=self.distribution, 
                                              zipfian_theta=self.zipfian_theta,
                                              orderedinserts=self.orderedinserts,
                                              min_field_length=self.min_field_length,
                                              max_field_length=self.max_field_length,
                                              field_count=self.field_count,
                                              value_type=self.value_type,
                                              key_type=self.key_type)
        
        if self.binding_type == BindingType.PYTHON:
            binding_class: Type[BindingBase] = get_binding_class(binding_name)
            self.bind_instance = binding_class(
                workload_gen=self.workload_gen,
                num_records=num_records,
                distribution=distribution,
                scan_size=scan_size,
                gpu_device=gpu_device
            )
            
        elif self.binding_type == BindingType.CPP:
            self.bind_instance = YCSBBridge(key_type="uint64", value_type=value_type, binding=binding_name)
            self.max_batch_size = max(param[1] if isinstance(param, list) else 1 for op, param in self.operations.items())
            

            self.bind_instance.initialize(dim=self.field_count, 
                                          max_batch_size=self.max_batch_size, 
                                          binding_config=json.dumps(self.binding_config), 
                                          hbm_gb=self.binding_config.get('hbm_gb', 0), 
                                          gpu_ids=gpu_device, 
                                          gpu_init_capacity=self.binding_config.get('gpu_init_capacity', 0), 
                                          gpu_max_capacity=self.binding_config.get('gpu_max_capacity', 0))

        self._load_data(load_data_output_file=load_data_output_file)

    def _load_data(self, load_data_output_file):
        
        start_key = None
        end_key = None
        if self.binding_type == BindingType.PYTHON:
            self.bind_instance.load_data(load_data_output_file=load_data_output_file, num_processes=self.generator_num_processes, start_key=start_key, end_key=end_key)
        elif self.binding_type == BindingType.CPP:
            if load_data_output_file:
                load_data_list = []
        
            total_keys, total_values = self.workload_gen.generate_load_data(self.num_records, start_key=start_key, end_key=end_key, num_processes=self.generator_num_processes)
            # flatten the total_values into 1D array
            total_values = total_values.flatten()
            assert len(total_values) == self.num_records * self.workload_gen.field_count
            # batch insert into the table
            load_max_batch_size = self.max_batch_size if self.max_batch_size > 1 else 1000
            for i in range(0, self.num_records, load_max_batch_size):
                batch_size = min(load_max_batch_size, self.num_records - i)
                h_keys = total_keys[i:i+batch_size]
                h_values = total_values[i*self.workload_gen.field_count:(i+batch_size)*self.workload_gen.field_count]
            
                self.bind_instance.multiset_for_loading(batch_size, h_keys, h_values)
            
            if load_data_output_file: 
                with open(load_data_output_file, 'w') as f:
                    json.dump(load_data_list, f, indent=4)
            

    def _load_operations(self):
        if not self.operations_file or not os.path.exists(self.operations_file):
            return None
            
        try:
            with open(self.operations_file, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                return None
                
            # First line contains metadata
            metadata_line = lines[0].strip()
            if not metadata_line.startswith('#'):
                print(f"Invalid operations file format: missing metadata")
                return None
                
            # Parse metadata
            metadata = {}
            metadata_parts = metadata_line[1:].split('|')
            for part in metadata_parts:
                key, value = part.split(':', 1)
                if key in ['num_records', 'scan_size', 'min_field_length', 'max_field_length']:
                    metadata[key] = int(value)
                elif key == 'orderedinserts':
                    metadata[key] = value.lower() == 'true'
                elif key == 'data_integrity':
                    metadata[key] = value.lower() == 'true'
                else:
                    metadata[key] = value
                    
            # Check if metadata matches current requirements
            if (metadata.get('num_records') == self.num_records           
                and metadata.get('distribution') == self.distribution        
                and metadata.get('scan_size') == self.scan_size              
                and metadata.get('orderedinserts') == self.orderedinserts    
                and metadata.get('data_integrity') == self.data_integrity    
                and metadata.get('min_field_length') == self.min_field_length 
                and metadata.get('max_field_length') == self.max_field_length 
                and metadata.get('workload_name') == self.workload_name
                ):
                
                # Parse operations data
                operations_data = {}
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split('|')
                    op_type = parts[0]
                    
                    if op_type not in operations_data:
                        operations_data[op_type] = []
                    
                    op_data = {}
                    for part in parts[1:]:
                        key, value = part.split(':', 1)
                        
                        if key in ['keys', 'key_list']:
                            # Parse list of keys
                            op_data[key] = [int(x) for x in value.split(',')]
                        elif key == 'value_list':
                            # Parse list of values
                            op_data[key] = [float(x) for x in value.split(',')]
                        elif key == 'values':
                            # Parse dict of field:value pairs
                            values_dict = {}
                            for field_value in value.split(','):
                                field, val = field_value.split(':', 1)
                                values_dict[field] = float(val)
                            op_data[key] = values_dict
                        elif key in ['batch_size']:
                            op_data[key] = int(value)
                        elif key in ['key', 'start_key', 'fieldkey']:
                            # Try to parse as int first, fallback to string
                            try:
                                op_data[key] = int(value)
                            except ValueError:
                                op_data[key] = value
                        elif key == 'value':
                            op_data[key] = float(value)
                        else:
                            op_data[key] = value
                    
                    operations_data[op_type].append(op_data)
                
                return operations_data
            else:
                print(f"Saved operations don't match requirements")
        except Exception as e:
            print(f"Error loading operations file: {e}")
        return None

    def _save_operations(self, operation_data):
        if not self.operations_file:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.operations_file), exist_ok=True)
            
            with open(self.operations_file, 'w') as f:
                # Write metadata as first line
                metadata_parts = [
                    f"num_records:{self.num_records}",
                    f"distribution:{self.distribution}",
                    f"scan_size:{self.scan_size}",
                    f"orderedinserts:{self.orderedinserts}",
                    f"data_integrity:{self.data_integrity}",
                    f"min_field_length:{self.min_field_length}",
                    f"max_field_length:{self.max_field_length}",
                    f"workload_name:{self.workload_name}"
                ]
                f.write('#' + '|'.join(metadata_parts) + '\n')
                
                # Write operations data line by line
                for op_type, op_list in operation_data.items():
                    for op_data in op_list:
                        line_parts = [op_type]
                        
                        for key, value in op_data.items():
                            if key in ['keys', 'key_list']:
                                # Save list of keys as comma-separated values
                                line_parts.append(f"{key}:{','.join(map(str, value))}")
                            elif key == 'value_list':
                                # Save list of values as comma-separated values
                                line_parts.append(f"{key}:{','.join(map(str, value))}")
                            elif key == 'values':
                                # Save dict as field:value,field:value format
                                values_str = ','.join([f"{field}:{val}" for field, val in value.items()])
                                line_parts.append(f"{key}:{values_str}")
                            else:
                                # Save other values directly
                                line_parts.append(f"{key}:{value}")
                        
                        f.write('|'.join(line_parts) + '\n')
                        
            print(f"Operations saved to {self.operations_file} (text format with {sum(len(op_list) for op_list in operation_data.values())} operations)")
        except Exception as e:
            print(f"Error saving operations file: {e}")

    def _generate_operations(self, operations_data: dict | None, num_ops: int):
        """Generate operations using multiple processes with Pool.starmap for optimal performance.
        
        Args:
            operations_data: Dictionary of operation data or None if no operations are loaded
            num_ops: Total number of operations to generate
            
        Returns:
            tuple: (operation_data, op_list)
        """
        result_operations_data = operations_data
        need_update = False
        ops_list = []
        num_processes = self.generator_num_processes
        if result_operations_data is None:
            result_operations_data = {}
        
        # Track the next available key for insert/multiset operations
        next_key_offset = 0
        # Collect all tasks for starmap
        starmap_args = []
        for op, param in self.operations.items():
            # each iteration is one type of operation (i.e. multiset, multiget, etc.)
            op_data = result_operations_data.get(op, None)
            if op_data is None:
                result_operations_data[op] = []
                op_data = result_operations_data[op]
            target_num_ops = int(param[0] * num_ops) if isinstance(param, list) else int(param * num_ops)
            remaining_num_ops = target_num_ops - len(op_data)
            if remaining_num_ops > 0:
                ops_list.extend([op] * len(op_data))
                need_update = True
                chunk_size = remaining_num_ops // num_processes
                remainder = remaining_num_ops % num_processes
                ranges = []
                for i in range(num_processes):
                    if len(ranges) == 0:
                        current_start = len(op_data)
                    else:
                        current_start = ranges[-1][1]
                    if current_start >= target_num_ops:
                        break
                    end_idx = current_start + chunk_size
                    if i < remainder:
                        end_idx += 1
                    ranges.append((current_start, end_idx))
                    
                assert sum([end_idx - start_idx for start_idx, end_idx in ranges]) == remaining_num_ops
                
                # Prepare arguments for starmap
                for start_idx, end_idx in ranges:
                    starmap_args.append((op, param, self.workload_gen, self.num_records, start_idx, end_idx, next_key_offset))
                
                # For insert and multiset operations, update the next_key_offset
                if op in [OperationFactory.get_operation('insert').name, OperationFactory.get_operation('multiset').name]:
                    if isinstance(param, list):
                        # multiset: batch_size * num_operations
                        batch_size = param[1]
                        next_key_offset += remaining_num_ops * batch_size
                    else:
                        # insert: single key per operation
                        next_key_offset += remaining_num_ops
            else:
                result_operations_data[op] = op_data[:target_num_ops]
                ops_list.extend([op] * target_num_ops)
        
        # Execute multiprocessing tasks using Pool.starmap
        if starmap_args:
            with mp.get_context('fork').Pool(processes=num_processes) as pool:
                results = pool.starmap(_generate_operation_worker, starmap_args)
            
            # Collect results
            for op_data, ops in results:
                for op_type in op_data:
                    result_operations_data[op_type].extend(op_data[op_type])
                ops_list.extend(ops)
            
        # Save the updated operations    
        if need_update:
            self._save_operations(result_operations_data)
            
        return result_operations_data, ops_list
    
    
    def run(self, num_ops=10_000, num_streams=0, warmup=False):
        total_time = 0
        
        # Calculate total operations including existing ones
        total_ops = 0
        for op, param in self.operations.items():
            op_count = int(param[0] * num_ops) if isinstance(param, list) else int(param * num_ops)
            if isinstance(param, list):
                total_ops += (op_count * param[1])
            else:
                total_ops += op_count
        
        
        
        if self.binding_type == BindingType.PYTHON:
            total_time, integrity, integrity_accuracy = self.run_py(num_ops, num_streams)
        elif self.binding_type == BindingType.CPP:
            total_time, integrity, integrity_accuracy = self.run_cpp(num_ops, num_streams)
        else:
            raise ValueError(f"Unsupported binding type: {self.binding_type}")
        
        
        if not warmup:
        
            throughput = total_ops / total_time
            avg_latency_per_query = total_time / total_ops * 1000  # Convert to milliseconds
            avg_latency_per_batch_op = total_time / num_ops * 1000
            
            
            # Prepare results
            results = {
                'workload': self.workload_name,
                'distribution': self.distribution,
                'num_records': self.num_records,
                'num_batch_ops': num_ops,
                'avg_latency_per_query_ms': avg_latency_per_query,
                'avg_latency_per_batch_op_ms': avg_latency_per_batch_op,
                'throughput': throughput,
                'total_time': total_time,
                'binding': self.binding_name,
                'binding_type': self.binding_type.name,
                'num_streams': num_streams,
                'field_count': self.field_count,
                'orderedinserts': self.orderedinserts,
                'data_integrity': self.data_integrity,
                'integrity': integrity,
                'integrity_accuracy': integrity_accuracy,
                'value_type': self.value_type
            }
            
            if self.value_type == "string":
                results['min_field_length'] = self.min_field_length
                results['max_field_length'] = self.max_field_length
            
            
            if self.distribution == 'zipfian':
                results['zipfian_theta'] = self.zipfian_theta
            
            for op, param in self.operations.items():
                if isinstance(param, list):
                    results[f"{op}_prob"] = param[0]
                    results[f'{op}_batch_size'] = param[1]
                else:
                    results[f"{op}_prob"] = param
                    
        
            # Print results
            print(f"\nWorkload {self.workload_name} Results:")
            print(f"Distribution: {self.distribution}")
            print(f"Total Operations: {total_ops}")
            print(f"Number of batches: {num_ops}")
            print(f"Average Latency per query: {results['avg_latency_per_query_ms']} ms")
            print(f"Average Latency per batch op: {results['avg_latency_per_batch_op_ms']} ms")
            print(f"Throughput: {results['throughput']} ops/sec")
            print(f"Integrity: {integrity}")
            print(f"Integrity Accuracy: {integrity_accuracy}")
            print(f"Total Time: {results['total_time']} seconds")
            # Write results to file if specified
            if self.output_file:
                with open(self.output_file, 'w') as f:
                    json.dump(results, f, indent=4)
                    print(f"\nResults have been written to {self.output_file}")
        
            return results 
    
    def run_cpp(self, num_ops=10_000, num_streams=0):
        print(f"Starting workload {self.operations} with {num_ops} operations...")
        if self.target_qps:
            print(f"Target QPS: {self.target_qps}")
        # Try to load operations from file
        loaded_operation_data = self._load_operations()
        all_operation_data, all_op_list = self._generate_operations(loaded_operation_data, num_ops)

        # Shuffle the combined op_list
        random.shuffle(all_op_list)
        
        workload_ops = []
        op_counters = {op: 0 for op in all_operation_data.keys()}
        
        for op in all_op_list:
            if isinstance(op, list):
                op_name, _ = op
                data = all_operation_data[op_name][op_counters[op_name]]
                
                if op_name == 'multiget':
                    workload_ops.append(CPPOperation(op='multiget', keys=data['keys']))
                elif op_name == 'multiset':
                    workload_ops.append(CPPOperation(op='multiset', keys=data['key_list'], values=data['value_list']))
                elif op_name == 'multiput':
                    workload_ops.append(CPPOperation(op='multiput', keys=data['key_list'], values=data['value_list']))
                else:
                    raise ValueError(f"Unsupported multi-operation for CPP binding: {op_name}. Only multiget, multiset, and multiput are supported.")
                
            else:
                data = all_operation_data[op][op_counters[op]]
                if op == 'read':
                    workload_ops.append(CPPOperation(op='read', keys=data['key']))
                else:
                    raise ValueError(f"Unsupported single-operation for CPP binding: {op}. Only read is supported.")
                op_counters[op] += 1
        
        if self.bind_instance.is_cuda():
            results = self.bind_instance.run_benchmark_cuda(workload_ops, num_streams=num_streams, data_integrity=self.data_integrity)
        else:
            results = self.bind_instance.run_benchmark_cpu(workload_ops, data_integrity=self.data_integrity)
            
        self.cleanup()
        
        return results.time_seconds, results.integrity, results.integrity_accuracy
        

    def run_py(self, num_ops=10_000, num_streams=0):
        print(f"Starting workload {self.operations} with {num_ops} operations...")
        if self.target_qps:
            print(f"Target QPS: {self.target_qps}")
        # Try to load operations from file
        loaded_operation_data = self._load_operations()
        all_operation_data, all_op_list = self._generate_operations(loaded_operation_data, num_ops)
        
        # Calculate total operations including existing ones
        total_ops = 0
        for op, param in self.operations.items():
            op_count = int(param[0] * num_ops) if isinstance(param, list) else int(param * num_ops)
            if isinstance(param, list):
                total_ops += (op_count * param[1])
            else:
                total_ops += op_count
        
        
        # Shuffle the combined op_list
        random.shuffle(all_op_list)
        
        op_counters = {op: 0 for op in all_operation_data.keys()}
        
        tasks = []
        
        # stream = cp.cuda.Stream()
        
        for op in all_op_list:
            if isinstance(op, list):
                op_name, _ = op
                data = all_operation_data[op_name][op_counters[op_name]]
                
                if op_name == 'multiget':
                    tasks.append((self.bind_instance.multiget, 
                                  (data['keys'], ), 
                                  (op_name, op_counters[op_name])))
                elif op_name == 'multiset':
                    tasks.append((self.bind_instance.multiset, 
                                  (data['key_list'], data['value_list']),
                                  (op_name, op_counters[op_name])))
                elif op_name == 'multiput':
                    # multiput uses the same multiset method but with existing keys
                    tasks.append((self.bind_instance.multiset, 
                                  (data['key_list'], data['value_list']),
                                  (op_name, op_counters[op_name])))
                else:
                    data = all_operation_data[op][op_counters[op]]
                    op_counters[op] += 1
                    if op == 'insert':
                        tasks.append((self.bind_instance.insert, (data['key'], data['values'])))
                    elif op == 'read':
                        tasks.append((self.bind_instance.read, (data['key'])))
                    elif op == 'update':
                        tasks.append((self.bind_instance.update, (data['key'], data['fieldkey'], 
                                               data['value'])))   
                    elif op == 'scan':
                        tasks.append((self.bind_instance.scan, (data['start_key'])))
                    else:
                        raise ValueError(f"Invalid operation: {op}")
                op_counters[op_name] += 1
                
        if num_streams is None:
            num_streams = len(tasks)

        executor = ThreadPoolExecutor(max_workers=num_streams,
                                      initializer=self.bind_instance.init_each_thread)
        
        # warm up
        for _ in range(num_streams):
            executor.submit(lambda: None).result()
            
        
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        binding_return_data = []
            
        print("Starting benchmark...")    
        # Start timing
        if self.data_integrity:
            start_event.record()
            futures = [executor.submit(fn, *args) for fn, args, _ in tasks]
            for future in as_completed_thread(futures):
                binding_return_data.append(future.result())
        else:
            start_event.record()
            futures = [executor.submit(fn, *args) for fn, args, _ in tasks]
            for future in as_completed_thread(futures):
                future.result()
            
            
        cp.cuda.Stream.null.synchronize()
        end_event.record()
        end_event.synchronize()
        
        executor.shutdown(wait=True)
        
        total_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000
        
        integrity = True
        
        # check data integrity
        print(f"Checking data integrity...")
        if self.data_integrity:
            cpu_count = mp.get_context('fork').cpu_count()
            executor = ProcessPoolExecutor(mp_context=mp.get_context('fork'), max_workers=cpu_count)
            futures = []
            for i, data in enumerate(binding_return_data):
                test_result: pd.DataFrame = self.bind_instance.results_postprocess(data)
                
                each_worker_check_size = test_result.shape[0] // cpu_count
                remainder = test_result.shape[0] % cpu_count
                
                current_idx = 0
                for j in range(cpu_count):
                    start_idx = current_idx
                    current_idx += each_worker_check_size
                    if j < remainder:
                        current_idx += 1
                    futures.append(executor.submit(check_rows_integrity_ycsb, test_result.iloc[start_idx:current_idx], self.workload_gen))

                if i % 10 == 0:
                    print(f"{i}/{num_ops} data integrity check submitted")
            i = 0
            for future in as_completed_process(futures):
                if not future.result():
                    integrity = False
                i += 1
                if i % 100 == 0:
                    print(f"{i}/{len(futures)} data integrity check passed")
            executor.shutdown(wait=True)
            print(f"Data integrity check passed")
        else:
            print(f"Data integrity type: {self.data_integrity} do not need to check")
        

        return total_time, integrity, -1

    def cleanup(self):
        self.bind_instance.cleanup()
        
def check_rows_integrity_ycsb(rows: pd.DataFrame, workload_gen):
    for _, row in rows.iterrows():
        key = row['key']
        values = workload_gen.build_values(key)
        for field, value in values.items():
            if row[field] != value:
                return False
    return True      
        



def generate_records_chunk(chunk_size, start_idx, process_id, workload_gen):
    """Worker function to generate a chunk of records"""
    keys = []
    values = []
    for i in range(start_idx, start_idx + chunk_size):
        key = workload_gen.build_key_name(i, 8)
        data = workload_gen.build_values(key)
        keys.append(key)
        for k, v in data.items():
            values.append(v)
            
        if len(keys) % 100000 == 0:
            print(f"Generated {len(keys)} keys and {len(values)} values")
            
    return keys, values