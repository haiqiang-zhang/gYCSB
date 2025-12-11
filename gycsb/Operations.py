import random
from .WorkloadGenerator import WorkloadGenerator, OperationFactory, Operation
    
@OperationFactory.register
class MultiGet(Operation):
    name = "multiget"
    description = "Get multiple records from the database"
    is_batch = True
        
    def gen_workload(self, start_idx, end_idx, batch_size, num_records, workload_gen: WorkloadGenerator, prob_not_in_original_records=0) -> list[dict]:
        operation_data = []
        for _ in range(start_idx, end_idx):
            
            keys = workload_gen.get_batch_key_names(batch_size, num_records, allow_duplicates=True, prob_not_in_original_records=prob_not_in_original_records)
            random.shuffle(keys)
            operation_data.append({
                'keys': keys,
                'batch_size': batch_size
            })
        return operation_data

@OperationFactory.register
class MultiSet(Operation):
    name = "multiset"
    description = "Set multiple records in the database"
    is_batch = True
    
    def gen_workload(self, start_idx, end_idx, batch_size, num_records, next_key_offset, workload_gen: WorkloadGenerator, prob_not_in_original_records=0) -> list[dict]:
        operation_data = []
        num_not_in_original_records = int(batch_size * prob_not_in_original_records)
        for i in range(start_idx, end_idx):
            # Generate new keys starting from num_records + next_key_offset
            # Each worker generates unique keys based on its operation index (i)
            keys_list = []
            base_key_index = num_records + next_key_offset + i * batch_size
            for j in range(num_not_in_original_records):
                key = workload_gen.build_key_name(base_key_index + j, 8)
                keys_list.append(key)
            
            keys_list.extend(workload_gen.get_batch_key_names(batch_size - num_not_in_original_records, num_records, allow_duplicates=True, prob_not_in_original_records=0))
            random.shuffle(keys_list)
            
            values_list = workload_gen.build_batch_values(batch_size, field_count=workload_gen.field_count, keys_list=keys_list)

            operation_data.append({
                'key_list': keys_list,
                'value_list': values_list
            })
        return operation_data

@OperationFactory.register
class MultiPut(Operation):
    name = "multiput"
    description = "Put multiple records in the database"
    is_batch = True
    
    def gen_workload(self, start_idx, end_idx, batch_size, num_records, workload_gen: WorkloadGenerator, prob_not_in_original_records=0) -> list[dict]:
        operation_data = []
        for _ in range(start_idx, end_idx):
            # Generate existing keys (like multiget) but also generate values for them (like multiset)
            keys_list = workload_gen.get_batch_key_names(batch_size, num_records, allow_duplicates=False, prob_not_in_original_records=prob_not_in_original_records)
            random.shuffle(keys_list)
            # Generate values for the existing keys
            values_list = workload_gen.build_batch_values(batch_size, field_count=workload_gen.field_count, keys_list=keys_list)

            operation_data.append({
                'key_list': keys_list,
                'value_list': values_list,
                'batch_size': batch_size
            })
            
        return operation_data

@OperationFactory.register
class Insert(Operation):
    name = "insert"
    description = "Insert a new record into the database"
    is_batch = False

@OperationFactory.register
class Read(Operation):
    name = "read"
    description = "Read a record from the database"
    is_batch = False


@OperationFactory.register
class Update(Operation):
    name = "update"
    description = "Update a record in the database"
    is_batch = False


@OperationFactory.register
class Scan(Operation):
    name = "scan"
    description = "Scan records from the database"
    is_batch = False


