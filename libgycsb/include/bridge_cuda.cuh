#pragma once
#include <string>
#include <vector>
#include <memory>
#include "binding_interface.h"
#include "bridge.h"
#include <cuda_runtime.h>

template<typename K, typename V>
class YCSBBridgeCUDA {
public:
    YCSBBridgeCUDA(const std::string& binding_name);
    ~YCSBBridgeCUDA();

    void initialize(uint64_t gpu_init_capacity, uint64_t gpu_max_capacity, uint32_t dim, uint32_t hbm_gb, std::vector<int> gpu_ids, uint64_t max_batch_size, const std::string& binding_config);

    void multiset_for_loading(uint32_t batch_size, const K* keys, const V* values, cudaStream_t stream = 0);

    BenchmarkResult run_benchmark(const std::vector<Operation<K, V>>& ops, uint64_t num_streams=1, bool data_integrity=true);

    void cleanup();

    // Static method to get available bindings for this K,V type combination
    static std::vector<std::string> getAvailableBindings();

private:
    std::shared_ptr<IBinding<K, V>> binding_;
    uint32_t dim_;
    uint32_t hbm_gb_;
    std::vector<int> gpu_ids_;
    uint64_t max_batch_size_;
    std::string binding_config_;
    
    std::pair<bool, double> verify_integrity_ycsb(const std::vector<Operation<K, V>>& stored_ops, const std::vector<V*>& d_values_out_list, const std::vector<bool*>& d_found_list);
};