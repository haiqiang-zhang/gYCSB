from benchmark.ycsb.YCSBController import YCSBController
from benchmark.ycsb.ConfigLoader import get_workload_config, get_binding_config
import json

def main():
    workload_name = "Multi_Set_16384"
    dims = [8, 16, 32]
    RESULTS_FILE = "/pub/nfs-data/zhaiqiang/autogpudb/benchmark/results/insert_and_evict_upsert_and_evict_kernel_with_io_core.json"
    binding_names = ["hkv_kernelopt"]
    workload_config = get_workload_config(workload_name)
    
    
    for binding in binding_names:
        binding_config = get_binding_config(binding)
        for dim in dims:
            controller = YCSBController(
                num_records=workload_config["num_records"],
                operations=workload_config["operations"],
                workload_name=workload_config["name"],
                distribution=workload_config["distribution"],
                zipfian_theta=workload_config["zipfian_theta"],
                orderedinserts=workload_config["orderedinserts"],
                data_integrity=workload_config["data_integrity"],
                min_field_length=workload_config["min_field_length"],
                max_field_length=workload_config["max_field_length"],
                field_count=dim,
                binding_type="cpp",
                binding_name=binding,
                output_file=f"{binding}_result.json",
                binding_config=binding_config,
                gpu_device=binding_config["gpu_ids"],
                generator_num_processes=50,
                value_type=binding_config["value_type"]
            )
            result = controller.run(num_ops=workload_config["ops"])
        

            with open(RESULTS_FILE, "r") as f:
                results = json.load(f)
            results["results"].append(result)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Dim {dim} results have been written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()