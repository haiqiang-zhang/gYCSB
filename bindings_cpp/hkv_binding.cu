#include "binding_registry.h"
#include "binding_interface.h"
#include "merlin_hashtable.cuh"
#include <iostream>
#include <stdexcept>
#include <string>
#include <memory>

#include <nlohmann/json.hpp>

#define HKV_CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    }                                                                   \
  } while (0)

template<typename K, typename V>
class HKVBaselineBinding : public IBinding<K, V> {
private:
    using HKVTable = nv::merlin::HashTable<K, V, uint64_t, nv::merlin::EvictStrategy::kLru>;
    using TableOptions = nv::merlin::HashTableOptions;

    std::unique_ptr<HKVTable> table_;
    uint32_t dim_;
    uint32_t max_batch_size_;
    uint64_t* d_scores_;
    std::string multiset_operation_;
    std::string multiget_operation_;

    K* d_evicted_keys_;
    V* d_evicted_values_;
    size_t* d_evicted_counter_;

    bool* d_accum_or_assigns_bools_;
public:
    HKVBaselineBinding();
    ~HKVBaselineBinding() override;
    
    bool initialize(const InitConfig& config) override;
    
    void cleanup() override;
    
    void multiset_for_loading(uint32_t batch_size,
                 const K* d_keys,
                 const V* d_values,
                 const CallContext& ctx = {}) override;
    void multiset(uint32_t batch_size,
                const K* d_keys,
                const V* d_values,
                const CallContext& ctx = {}) override;                 
    void multiput(uint32_t batch_size,
        const K* d_keys,
        const V* d_values,
        const CallContext& ctx = {}) override;
    
    void multiget(uint32_t batch_size,
                 const K* d_keys,
                 V* d_values_out,
                 bool* d_found,
                 const CallContext& ctx = {}) override;


    void insert_and_evict(uint32_t batch_size,
                    const K* d_keys,
                    const V* d_values,
                    const CallContext& ctx);

    void accum_or_assign(uint32_t batch_size,
        const K* d_keys,
        const V* d_values,
        const CallContext& ctx = {});

    void find_or_insert(uint32_t batch_size,
        const K* d_keys,
        V* d_values_out,
        bool* d_found,
        const CallContext& ctx = {});

    void find(uint32_t batch_size,
        const K* d_keys,
        V* d_values_out,
        bool* d_found,
        const CallContext& ctx = {});

};

template<typename K, typename V>
HKVBaselineBinding<K, V>::HKVBaselineBinding() 
    : table_(nullptr), dim_(0), max_batch_size_(0) {
}

template<typename K, typename V>
HKVBaselineBinding<K, V>::~HKVBaselineBinding() {
}

template<typename K, typename V>
bool HKVBaselineBinding<K, V>::initialize(const InitConfig& config) {
    // check if gpu_ids is only one
    if (config.gpu_ids.size() != 1) {
        throw std::runtime_error("HKV binding only supports one GPU");
    }

    int gpu_id = config.gpu_ids[0];

    // set device
    HKV_CUDA_CHECK(cudaSetDevice(gpu_id));
    nlohmann::json config_json = nlohmann::json::parse(config.additional_config);

    if (config_json.contains("multiset_operation") && !config_json["multiset_operation"].is_null()) {
        multiset_operation_ = config_json["multiset_operation"].get<std::string>();
    }

    if (config_json.contains("multiget_operation") && !config_json["multiget_operation"].is_null()) {
        multiget_operation_ = config_json["multiget_operation"].get<std::string>();
    }


    // Store dimension
    dim_ = config.dim;
    
    // Initialize HKV table options
    TableOptions options;
    options.init_capacity = config.init_capacity;
    options.max_capacity = config.max_capacity;
    options.dim = config.dim;
    options.max_hbm_for_vectors = nv::merlin::GB(config.hbm_gb);
    options.io_by_cpu = false;
    options.device_id = gpu_id;

    // print options
    std::cout << "HKV baseline options: " << std::endl;
    std::cout << "  init_capacity: " << options.init_capacity << std::endl;
    std::cout << "  max_capacity: " << options.max_capacity << std::endl;
    std::cout << "  dim: " << options.dim << std::endl;
    std::cout << "  hbm_gb: " << config.hbm_gb << std::endl;
    std::cout << "  io_by_cpu: " << options.io_by_cpu << std::endl;
    std::cout << "  device_id: " << options.device_id << std::endl;
    std::cout << "  multiset_operation: " << multiset_operation_ << std::endl;
    std::cout << "  multiget_operation: " << multiget_operation_ << std::endl;

    
    // Create and initialize HKV table
    std::cout << "Creating HKV table" << std::endl;
    table_ = std::make_unique<HKVTable>();
    table_->init(options);
    std::cout << "HKV table created" << std::endl;
    
    // Store max batch size (no longer need to pre-allocate buffers)
    max_batch_size_ = config.max_batch_size;

    // initialize d_scores
    uint64_t* h_scores = new uint64_t[max_batch_size_];
    for (uint32_t i = 0; i < max_batch_size_; i++) {
        h_scores[i] = i;
    }
    HKV_CUDA_CHECK(cudaMalloc(&d_scores_, max_batch_size_ * sizeof(uint64_t)));
    HKV_CUDA_CHECK(cudaMemcpy(d_scores_, h_scores, max_batch_size_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
    delete[] h_scores;
    


    // initialize evicted related buffers
    HKV_CUDA_CHECK(cudaMalloc(&d_evicted_keys_, max_batch_size_ * sizeof(K)));
    HKV_CUDA_CHECK(cudaMalloc(&d_evicted_values_, max_batch_size_ * dim_ * sizeof(V)));
    HKV_CUDA_CHECK(cudaMalloc(&d_evicted_counter_, sizeof(size_t)));
    HKV_CUDA_CHECK(cudaMemset(d_evicted_counter_, 0, sizeof(size_t)));


    if (multiset_operation_ == "accum_or_assign") {
        HKV_CUDA_CHECK(cudaMalloc(&d_accum_or_assigns_bools_, max_batch_size_ * sizeof(bool)));
        HKV_CUDA_CHECK(cudaMemset(d_accum_or_assigns_bools_, 1, max_batch_size_ * sizeof(bool)));
    }


    std::cout << "HKV binding initialized successfully" << std::endl;
    return true;
}

template<typename K, typename V>
void HKVBaselineBinding<K, V>::cleanup() {
    table_.reset();
    std::cout << "HKV binding cleaned up" << std::endl;
}

template<typename K, typename V>
void HKVBaselineBinding<K, V>::multiset_for_loading(uint32_t batch_size,
                               const K* d_keys,
                               const V* d_values,
                               const CallContext& ctx) {

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
    
    // Keys and values are already on device, directly perform insert operation
    table_->insert_or_assign(batch_size, d_keys, d_values, nullptr, stream, false);

    cudaStreamSynchronize(stream);
    

}


template<typename K, typename V>
void HKVBaselineBinding<K, V>::multiset(uint32_t batch_size,
                               const K* d_keys,
                               const V* d_values,
                               const CallContext& ctx) {
    
    if (multiset_operation_ == "insert_and_evict") {
        insert_and_evict(batch_size, d_keys, d_values, ctx);
    } else if (multiset_operation_ == "accum_or_assign") {
        accum_or_assign(batch_size, d_keys, d_values, ctx);
    } else {
        throw std::runtime_error("Invalid multiset like operation: " + multiset_operation_);
    }
}


template<typename K, typename V>
void HKVBaselineBinding<K, V>::multiput(uint32_t batch_size,
                               const K* d_keys,
                               const V* d_values,
                               const CallContext& ctx) {
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
    // Keys and values are already on device, directly perform insert operation
    table_->assign(batch_size, d_keys, d_values, nullptr, stream, true);
    
    cudaStreamSynchronize(stream);
}


template<typename K, typename V>
void HKVBaselineBinding<K, V>::multiget(uint32_t batch_size,
                               const K* d_keys,
                               V* d_values_out,
                               bool* d_found,
                               const CallContext& ctx) {
    if (multiget_operation_ == "find") {
        find(batch_size, d_keys, d_values_out, d_found, ctx);
    } else if (multiget_operation_ == "find_or_insert") {
        find_or_insert(batch_size, d_keys, d_values_out, d_found, ctx);
    } else {
        throw std::runtime_error("Invalid multiget operation: " + multiget_operation_);
    }

}

template<typename K, typename V>
void HKVBaselineBinding<K, V>::insert_and_evict(uint32_t batch_size,
                               const K* d_keys,
                               const V* d_values,
                               const CallContext& ctx) {

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
    
    table_->insert_and_evict(batch_size, d_keys, d_values, nullptr, d_evicted_keys_, d_evicted_values_, nullptr, d_evicted_counter_, stream, false);

    cudaStreamSynchronize(stream);
}



template<typename K, typename V>
void HKVBaselineBinding<K, V>::accum_or_assign(uint32_t batch_size,
    const K* d_keys,
    const V* d_values,
    const CallContext& ctx) {

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
    
    table_->accum_or_assign(batch_size, d_keys, d_values, d_accum_or_assigns_bools_, nullptr, stream, false);

    cudaStreamSynchronize(stream);
}


template<typename K, typename V>
void HKVBaselineBinding<K, V>::find(uint32_t batch_size,
                               const K* d_keys,
                               V* d_values_out,
                               bool* d_found,
                               const CallContext& ctx) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);

    table_->find(batch_size, d_keys, d_values_out, d_found, nullptr, stream);
    
    cudaStreamSynchronize(stream);

}

template<typename K, typename V>
void HKVBaselineBinding<K, V>::find_or_insert(uint32_t batch_size,
                               const K* d_keys,
                               V* d_values_out,
                               bool* d_found,
                               const CallContext& ctx) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);

    table_->find_or_insert(batch_size, d_keys, d_values_out, nullptr, stream, false);
    
    cudaStreamSynchronize(stream);

}


// Register the HKV bindings with the registry using simplified macros
using HKVBaselineBinding_u64d = HKVBaselineBinding<uint64_t, double>;
REGISTER_CUDA_BINDING(uint64_t, double, HKVBaselineBinding_u64d, "hkv_baseline");

using HKVBaselineBinding_u64f = HKVBaselineBinding<uint64_t, float>;
REGISTER_CUDA_BINDING(uint64_t, float, HKVBaselineBinding_u64f, "hkv_baseline");

// using HKVBaselineBinding_u64u64 = HKVBaselineBinding<uint64_t, uint64_t>;
// REGISTER_CUDA_BINDING(uint64_t, uint64_t, HKVBaselineBinding_u64u64, "hkv_baseline"); 