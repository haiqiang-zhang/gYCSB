from benchmark.ycsb.YCSBController import YCSBController
from benchmark.ycsb.ConfigLoader import get_workload_config, get_binding_config
import json

def main():
    ops_type = "Get"
    workload_names = [
                      f"Multi_{ops_type}_1024", 
                      f"Multi_{ops_type}_2048", 
                      f"Multi_{ops_type}_4096", 
                      f"Multi_{ops_type}_8192", 
                      f"Multi_{ops_type}_16384", 
                      f"Multi_{ops_type}_32768", 
                      f"Multi_{ops_type}_65536", 
                      f"Multi_{ops_type}_131072"
                      ]
    RESULTS_FILE = "/pub/nfs-data/zhaiqiang/autogpudb/benchmark/results/find_hybird_read_thread.json"
    binding_names = ["hkv_baseline", "hkv_kernelopt"]
    
    
    for workload_name in workload_names:
        workload_config = get_workload_config(workload_name)
        for b in binding_names:
            binding_config = get_binding_config(b)
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
                field_count=workload_config["field_count"],
                binding_type="cpp",
                binding_name=b,
                output_file=f"{b}_result.json",
                binding_config=binding_config,
                gpu_device=binding_config["gpu_ids"],
                generator_num_processes=50,
                value_type=binding_config["value_type"]
            )
            result = controller.run(num_ops=workload_config["ops"])
            

            with open(RESULTS_FILE, "r") as f:
                results = json.load(f)
            results.append(result)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Workload {workload_name} results have been written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()