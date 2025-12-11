import random
import string
import numpy as np
from functools import partial
from gycsb.ZipfianGenerator import ZipfianGenerator
import pyarrow as pa
import multiprocessing as mp
from abc import ABC, abstractmethod

class WorkloadGenerator:
    def __init__(self, 
                 data_integrity: bool = True, 
                 distribution="zipfian",
                 zipfian_theta=0,
                 field_count=10, 
                 field_prefix="field",
                 orderedinserts=False,
                 min_field_length=128,
                 max_field_length=512,
                 value_type: str = "double",
                 key_type: str = "np_int64",
                 scan_size=100):
        
        self.data_integrity = data_integrity
        self.distribution = distribution
        self.min_field_length = min_field_length
        self.max_field_length = max_field_length
        self.field_count = field_count
        self.field_prefix = field_prefix
        self.orderedinserts = orderedinserts
        self.fields = [self.field_prefix + str(i) for i in range(self.field_count)]
        self.scan_size = scan_size
        # Initialize ZipfianGenerator for reuse
        self._zipfian_generator = None
        self.zipfian_theta = zipfian_theta
        self.value_type = value_type
        self.key_type = key_type
        
    def _random_number_generator(self, min_val, max_val, zipfian_constant=0):
        if self.distribution == 'uniform':
            return random.randint(min_val, max_val)
        elif self.distribution == 'zipfian':
            if self._zipfian_generator is None or self._zipfian_generator.items != (max_val - min_val + 1):
                self._zipfian_generator = ZipfianGenerator(min_val, max_val, zipfian_constant=zipfian_constant)
            return self._zipfian_generator.next_value()
        else:
            raise ValueError("Unsupported distribution")
        
    def get_key(self, num_records):
        """
        Generate a key number using uniform or zipfian distribution.
        """
        return self._random_number_generator(0, num_records - 1, self.zipfian_theta)
    
    def get_key_range(self, start_idx, end_idx):
        """
        Generate a key number using uniform or zipfian distribution.
        """
        return self._random_number_generator(start_idx, end_idx - 1, self.zipfian_theta)
        
    def field_length_generator(self):
        """
        Generate a field length using uniform or zipf distribution.
        """
        return self._random_number_generator(self.min_field_length, self.max_field_length, self.zipfian_theta)

    def _get_batch_key_names_worker(self, batch_size, num_records, start_idx=None, end_idx=None, allow_duplicates=False):
        keys = []
        if not allow_duplicates:
            while len(keys) < batch_size:
                if start_idx is not None and end_idx is not None:
                    keynum = self.get_key_range(start_idx, end_idx)
                else:
                    keynum = self.get_key(num_records)
                key = self.build_key_name(keynum, 8)
                if key not in keys:
                    keys.append(key)
                
                assert keynum < num_records
        else:    
            for _ in range(batch_size):
                if start_idx is not None and end_idx is not None:
                    keynum = self.get_key_range(start_idx, end_idx)
                else:
                    keynum = self.get_key(num_records)
                key = self.build_key_name(keynum, 8)
                keys.append(key)
                
        keys = sorted(keys)
        if self.key_type == 'np_int64':
            return np.array(keys, dtype=np.int64)
        elif self.key_type == 'np_int32':
            return np.array(keys, dtype=np.int32)
        elif self.key_type == 'pa_int64':
            return pa.array(keys, type=pa.int64())
        elif self.key_type == 'pa_int32':
            return pa.array(keys, type=pa.int32())
        elif self.key_type == 'int':
            return keys
        else:
            raise ValueError(f"Unsupported key_type: {self.key_type}")
    
    def get_batch_key_names(self, batch_size, num_records, num_processes=1, allow_duplicates=False, prob_not_in_original_records=0):
        
        if not allow_duplicates and batch_size > num_records:
            raise ValueError(f"Batch size {batch_size} is greater than number of records {num_records}")
        
        if num_processes == 1:
            if prob_not_in_original_records > 0:
                found_batch_size = int(batch_size * (1 - prob_not_in_original_records))
                not_found_batch_size = batch_size - found_batch_size
                found_key = self._get_batch_key_names_worker(found_batch_size, num_records, allow_duplicates=allow_duplicates)
                not_found_key = self._get_batch_key_names_worker(not_found_batch_size, num_records, start_idx=num_records, end_idx=num_records + batch_size, allow_duplicates=allow_duplicates)
                if self.key_type.startswith('np_'):
                    return np.concatenate([found_key, not_found_key])
                elif self.key_type.startswith('pa_'):
                    return pa.concat_arrays([found_key, not_found_key])
                else:
                    return found_key + not_found_key
            else:
                return self._get_batch_key_names_worker(batch_size, num_records, allow_duplicates=allow_duplicates)
        if prob_not_in_original_records > 0:
            found_batch_size = int(batch_size * (1 - prob_not_in_original_records))
            not_found_batch_size = batch_size - found_batch_size
            
            found_chunk_size = found_batch_size // num_processes
            found_remaining = found_batch_size % num_processes
            
            not_found_chunk_size = not_found_batch_size // num_processes
            not_found_remaining = not_found_batch_size % num_processes
            
            # Split key ranges for each process to avoid duplicates across processes
            found_range_size = num_records // num_processes
            not_found_range_size = batch_size // num_processes
            
            with mp.get_context('fork').Pool(processes=num_processes) as pool:
                keys = []
                # Generate found keys (within num_records range), each process gets a distinct range
                # Args: (batch_size, num_records, start_idx, end_idx, allow_duplicates)
                found_args = [
                    (found_chunk_size + (1 if i < found_remaining else 0), num_records, 
                     found_range_size * i, found_range_size * (i + 1) if i < num_processes - 1 else num_records, allow_duplicates)
                    for i in range(num_processes)
                ]
                found_results = pool.starmap(self._get_batch_key_names_worker, found_args)
                for result in found_results:
                    keys.extend(result)
                
                # Generate not-found keys (outside num_records range), each process gets a distinct range
                not_found_args = [
                    (not_found_chunk_size + (1 if i < not_found_remaining else 0), num_records,
                     num_records + not_found_range_size * i, num_records + not_found_range_size * (i + 1) if i < num_processes - 1 else num_records + batch_size,
                     allow_duplicates)
                    for i in range(num_processes)
                ]
                not_found_results = pool.starmap(self._get_batch_key_names_worker, not_found_args)
                for result in not_found_results:
                    keys.extend(result)
            return sorted(keys)
        else:
            chunk_size = batch_size // num_processes
            remaining = batch_size % num_processes
            
            # Split key range for each process to avoid duplicates across processes
            range_size = num_records // num_processes
            
            with mp.get_context('fork').Pool(processes=num_processes) as pool:
                keys = []
                # Args: (batch_size, num_records, start_idx, end_idx, allow_duplicates)
                args = [
                    (chunk_size + (1 if i < remaining else 0), num_records,
                     range_size * i, range_size * (i + 1) if i < num_processes - 1 else num_records,
                     allow_duplicates)
                    for i in range(num_processes)
                ]
                results = pool.starmap(self._get_batch_key_names_worker, args)
                for result in results:
                    keys.extend(result)
            return sorted(keys)
    
    def build_key_name(self, keynum, zeropadding):
        """
        Generate a formatted key string, e.g., 'user000001'.

        :param keynum: The numeric part of the key, e.g., 1, 2, 3...
        :param zeropadding: Number of zeroes to pad the key number with.
        :param orderedinserts: If False, hash the keynum to randomize key distribution.
        :return: A formatted key string like 'user000005'.
        """
        result = None
        if not self.orderedinserts:
            keynum = fnvhash64(keynum)
        if self.key_type in ['np_int64', 'np_int32', 'pa_int64', 'pa_int32', 'int']:
            result = keynum
        else:
            value = str(keynum)
            fill = zeropadding - len(value)
            prekey = "user" + ("0" * max(fill, 0))
            result = prekey + value
            
        return result

    def build_deterministic_value(self, key, fieldkey):
        
        if self.value_type == 'string':
            size = self.field_length_generator()
            sb = str(key) + ':' + fieldkey
            while len(sb) < size:
                sb += ':' + str(hash(sb))
            return sb[:size]
        
        elif self.value_type == 'double':
            fieldkeyIndex = self.fields.index(fieldkey)
            return np.float64((key+fieldkeyIndex) % (1 << 53))
        elif self.value_type == 'float':
            fieldkeyIndex = self.fields.index(fieldkey)
            return np.float32((key+fieldkeyIndex) % (1 << 24))
        else:
            raise ValueError(f"Unsupported value_type: {self.value_type}")

    def random_value(self, size=None, encoding='utf-8'):
        if self.value_type == 'string':
            result = ''
            while True:
                result += random.choice(string.printable)
                if len(result.encode(encoding)) >= size:
                    if len(result.encode(encoding)) == size:
                        return result
                    else:
                        result = result[:-1]
                        continue

        elif self.value_type == 'double':
            return np.float64(np.random.uniform(-1e100, 1e100)) 

        elif self.value_type == 'float':
            return np.float32(np.random.uniform(-1e38, 1e38))   

        elif self.value_type == 'int':
            return np.int32(np.random.randint(-2**31, 2**31))   

        elif self.value_type == 'long':
            return np.int64(np.random.randint(-2**63, 2**63))   

        else:
            raise ValueError(f"Unsupported value_type: {self.value_type}")
                
    

    def build_single_value(self, key=None):
        if self.data_integrity and key is None:
            raise ValueError("Key is required for YCSB data integrity")
        fieldkey = random.choice(self.fields)
        if self.data_integrity:
            data = self.build_deterministic_value(key, fieldkey)
        else:
            length = self.field_length_generator()
            data = self.random_value(length)
        return fieldkey, data
    
    def build_values(self, key=None):
        values = {}
        for field in self.fields:
            if self.data_integrity:
                values[field] = self.build_deterministic_value(key if key is not None else "", field)
            else:
                length = self.field_length_generator()
                values[field] = self.random_value(length)
        return values
    
    def build_batch_values(self, batch_size, field_count=1, keys_list=None):
        if field_count == 1:
            if keys_list is not None:
                values = [{'key': keys_list[i]} | self.build_single_value() for i in range(batch_size)]
            else:
                values = [self.build_single_value() for _ in range(batch_size)]
            return np.array(values, dtype=np.str_, order='C')
        elif field_count == len(self.fields):
            # Create arrays for each field
            # arrays = {}
            # if keys_list is not None:
            #     arrays['key'] = pa.array(keys_list, type=pa.int64())
            # for field in self.fields:
            #     v = [self.build_single_value()[1] for _ in range(batch_size)]
            #     arrays[field] = pa.array(v, type=pa.string())
            
            # # Return pyarrow table
            # return pa.table(arrays)
            batch_values = []
            for i in range(batch_size):
                values = self.build_values(keys_list[i])
                for _, v in values.items():
                    batch_values.append(v)
            return batch_values
        else:
            raise NotImplementedError(f"Multi-field values are not implemented for field_count: {field_count}")
    
    
    def generate_records_chunk_worker(self, chunk_size, start_idx):
        """Worker function to generate a chunk of records"""
        keys = []
        values = []
        for i in range(start_idx, start_idx + chunk_size):
            key = self.build_key_name(i, 8)
            value_dict = self.build_values(key)
            keys.append(key)
            values.append(list(value_dict.values()))
            
            
                
            if len(keys) % 100000 == 0:
                print(f"Generated {len(keys)} keys and {len(values)} values")
    
        return keys, values
    
    def generate_load_data(self, num_records, start_key=None, end_key=None, num_processes=64):
        start_key = None
        end_key = None
    
        if start_key is not None and end_key is not None:
            assert end_key - start_key == num_records
        
        # Calculate chunk size for each process
        chunk_size = num_records // num_processes
        remaining = num_records % num_processes
        
        if self.key_type == 'int':
            total_keys = []
        elif self.key_type == 'np_int64':
            total_keys = np.empty(num_records, dtype=np.int64)
        elif self.key_type == 'np_int32':
            total_keys = np.empty(num_records, dtype=np.int32)
        elif self.key_type == 'pa_int64':
            total_keys = np.empty(num_records, dtype=pa.int64())
        elif self.key_type == 'pa_int32':
            total_keys = np.empty(num_records, dtype=pa.int32())
        else:
            raise ValueError(f"Unsupported key_type: {self.key_type}")
        
        
        if self.value_type == 'double':
            total_values = np.empty((num_records, self.field_count), dtype=np.float64)
        elif self.value_type == 'float':
            total_values = np.empty((num_records, self.field_count), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported value_type: {self.value_type}")
        if num_processes == 1:
            keys, values = self.generate_records_chunk_worker(num_records, 0)
            total_keys[:] = keys
            total_values[:] = values

        else:
            with mp.get_context('fork').Pool(processes=num_processes) as pool:
                generate_chunk = partial(self.generate_records_chunk_worker)
                
                chunk_args = []
                current_idx = 0
                if start_key is not None and end_key is not None:
                    current_idx = start_key
                    
                for _ in range(num_processes):
                    chunk_args.append((chunk_size, current_idx))
                    current_idx += chunk_size
                
                chunks = pool.starmap(generate_chunk, chunk_args)
            
                print(f"Starting data merging from {len(chunks)} chunks...")
                # More efficient merging - directly populate the arrays
                current_idx = 0
                for chunk_idx, (chunk_keys, chunk_values) in enumerate(chunks):
                    chunk_key_len = len(chunk_keys)
                    chunk_value_len = len(chunk_values)
                    assert chunk_key_len == chunk_value_len
                    
                    total_keys[current_idx:current_idx + chunk_key_len] = chunk_keys
                    total_values[current_idx:current_idx + chunk_value_len] = chunk_values
                    
                    current_idx += chunk_key_len
                    
                    if chunk_idx % 10 == 0:
                        print(f"Merged chunk {chunk_idx+1}/{len(chunks)}")
        
            if remaining > 0:
                keys, values = self.generate_records_chunk_worker(remaining, current_idx)
                
                total_keys[current_idx:current_idx+remaining] = keys
                total_values[current_idx:current_idx+remaining] = values
                
        assert len(total_keys) == num_records
        assert len(total_values) == num_records
        
        return total_keys, total_values
        
def fnvhash64(val):
    """
    FNV-1a 64-bit hash function.
    from http://en.wikipedia.org/wiki/Fowler_Noll_Vo_hash
    """
    FNV_OFFSET_BASIS_64 = 14695981039346656037
    FNV_PRIME_64 = 1099511628211

    hashval = FNV_OFFSET_BASIS_64
    for _ in range(8):
        octet = val & 0xff
        val = val >> 8
        hashval = hashval ^ octet
        hashval = (hashval * FNV_PRIME_64) & 0xFFFFFFFFFFFFFFFF
    return abs(hashval)



class Operation(ABC):
    name: str
    description: str
    is_batch: bool
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, description={self.description}, is_batch={self.is_batch})"
    
    def __str__(self):
        return self.__repr__()

class OperationFactory:
    OPERATIONS: dict[str, Operation] = {}

    @classmethod
    def register(cls, op_cls):
        cls.OPERATIONS[op_cls.name] = op_cls
        return op_cls
        

    @staticmethod
    def get_operation(name: str) -> Operation:
        if name not in OperationFactory.OPERATIONS:
            raise ValueError(f"Operation {name} not found")
        return OperationFactory.OPERATIONS[name]()

    @staticmethod
    def get_operations() -> list[Operation]:
        return list(OperationFactory.OPERATIONS.values())
    
    
    @staticmethod
    def get_operations_details() -> dict[str, str]:
        return [str(op()) for op in OperationFactory.get_operations()]



def _generate_operation_worker(op, param, workload_gen, num_records, start_idx, end_idx, next_key_offset=0):
    """
    Worker function for Pool.starmap - generates operations efficiently in parallel processes.
    
    Args:
        op: Operation type (e.g., 'read', 'update', 'multiget', 'multiset')
        param: Operation parameters (proportion or [proportion, batch_size])
        workload_gen: WorkloadGenerator instance
        num_records: Total number of records in the database
        start_idx: Starting index for this worker's chunk
        end_idx: Ending index for this worker's chunk
        next_key_offset: Offset for generating new keys (for insert/multiset operations)
    
    Returns:
        tuple: (operation_data, op_list) containing generated operations
    """
    
    operation_data = {op: []}
    op_list = []
    if isinstance(param, list):
        batch_size = param[1]
        prob_not_in_original_records = 0
        if len(param) >= 3:
            prob_not_in_original_records = param[2]
        if op == 'multiget':
            operation_data['multiget'].extend(OperationFactory.get_operation(op).gen_workload(start_idx, end_idx, batch_size, num_records, workload_gen, prob_not_in_original_records))
                
        elif op == 'multiset':
            operation_data['multiset'].extend(OperationFactory.get_operation(op).gen_workload(start_idx, end_idx, batch_size, num_records, next_key_offset, workload_gen, prob_not_in_original_records))
                
        elif op == 'multiput':
            operation_data['multiput'].extend(OperationFactory.get_operation(op).gen_workload(start_idx, end_idx, batch_size, num_records, workload_gen, prob_not_in_original_records))
                
        op_list.extend([[op, batch_size]] * (end_idx - start_idx))
    else:
        if op == 'insert':
            for i in range(start_idx, end_idx):
                key_index = num_records + next_key_offset + i
                operation_data['insert'].append({
                    'key': workload_gen.build_key_name(key_index, 8),
                    'values': workload_gen.build_values(key=key_index)
                })
        elif op == 'read':
            for _ in range(start_idx, end_idx):
                operation_data['read'].append({
                    'key': workload_gen.build_key_name(workload_gen.get_key(num_records), 8)
                })
        elif op == 'update':
            for _ in range(start_idx, end_idx):
                keynum = workload_gen.get_key(num_records)
                key = workload_gen.build_key_name(keynum, 8)
                fieldkey, value = workload_gen.build_single_value(key=key)
                operation_data['update'].append({
                    'key': key,
                    'fieldkey': fieldkey,
                    'value': value
                })
        elif op == 'scan':
            for _ in range(start_idx, end_idx):
                operation_data['scan'].append({
                    'start_key': workload_gen.build_key_name(workload_gen.get_key(num_records), 8)
                })
        op_list.extend([op] * (end_idx - start_idx))
    
    # Return results directly instead of putting in queue
    return operation_data, op_list