#include "binding_registry.h"
#include "binding_interface.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <string>
#include <nlohmann/json.hpp>
#include "mlkv_plus.cuh"
#include <memory>

template<typename K, typename V>
class MLKVPlusBinding : public IBinding<K, V> {
private:
    std::unique_ptr<mlkv_plus::DB<K, V>> storage_;
    mlkv_plus::StorageConfig config_;
    uint32_t dim_;
    
public:
    MLKVPlusBinding();
    ~MLKVPlusBinding() override;
    
    bool initialize(const InitConfig& config) override;
    
    void cleanup() override;
    
    void multiset(uint32_t batch_size,
                 const K* d_keys,
                 const V* d_values,
                 const CallContext& ctx = {}) override;
    
    void multiget(uint32_t batch_size,
                 const K* h_keys,
                 V* d_values_out,
                 bool* h_found,
                 const CallContext& ctx = {}) override;


    void multiset_for_loading(uint32_t batch_size,
                              const K* d_keys,
                              const V* d_values,
                              const CallContext& ctx = {}) override;


    void get(const K* h_keys,
             V* d_values_out,
             bool* h_found,
             const CallContext& ctx = {}) override;
    
}; 


template<typename K, typename V>
MLKVPlusBinding<K, V>::MLKVPlusBinding() 
    : storage_(nullptr), dim_(0), config_{} {
        std::cout << "MLKVPlusBinding constructor" << std::endl;
}

template<typename K, typename V>
MLKVPlusBinding<K, V>::~MLKVPlusBinding() {
}

template<typename K, typename V>
bool MLKVPlusBinding<K, V>::initialize(const InitConfig& config) {
    try {

        // check if gpu_ids is only one
        if (config.gpu_ids.size() != 1) {
            throw std::runtime_error("MLKV+ binding only supports one GPU");
        }

        int gpu_id = config.gpu_ids[0];

        // Store dimension
        dim_ = config.dim;
        
        // Configure MLKV Plus storage
        config_.hkv_init_capacity = config.init_capacity;
        config_.hkv_max_capacity = config.max_capacity;
        config_.dim = config.dim;
        config_.max_hbm_for_vectors_gb = config.hbm_gb;
        config_.hkv_io_by_cpu = false;
        config_.gpu_id = gpu_id;
        config_.create_if_missing = true;
        config_.max_batch_size = config.max_batch_size;
        nlohmann::json config_json = nlohmann::json::parse(config.additional_config);
        // Set RocksDB path based on additional config or use default
        if (config_json.contains("rocksdb_path") && !config_json["rocksdb_path"].is_null()) {
            config_.rocksdb_path = config_json["rocksdb_path"].get<std::string>();
        } else {
            config_.rocksdb_path = "/tmp/mlkvplus_ycsb_db_" + std::to_string(gpu_id);
        }
        config_.enable_gds_log = config_json["enable_gds_log"].get<bool>();
        config_.disableWAL = config_json["disableWAL"].get<bool>();
        config_.enable_gds_get_from_sst = config_json["enable_gds_get_from_sst"].get<bool>();
        config_.force_skip_memtable = config_json["force_skip_memtable"].get<bool>();
        config_.rocksdb_use_direct_reads = config_json["rocksdb_use_direct_reads"].get<bool>();
        
        // Create and initialize storage
        storage_ = std::make_unique<mlkv_plus::DB<K, V>>(config_);
        
        auto result = storage_->initialize();
        if (result != mlkv_plus::OperationResult::SUCCESS) {
            std::cerr << "Failed to initialize MLKV Plus storage" << std::endl;
            return false;
        }
        
        std::cout << "MLKV Plus binding initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "MLKV Plus binding initialization failed: " << e.what() << std::endl;
        return false;
    }
}

template<typename K, typename V>
void MLKVPlusBinding<K, V>::cleanup() {
    if (storage_) {
        storage_->cleanup();
        storage_.reset();
    }
    std::cout << "MLKV Plus binding cleaned up" << std::endl;
}

template<typename K, typename V>
void MLKVPlusBinding<K, V>::multiset(uint32_t batch_size,
                                    const K* d_keys,
                                    const V* d_values,
                                    const CallContext& ctx) {
    if (!storage_) {
        throw std::runtime_error("MLKV Plus binding not initialized");
    }

    // Perform multiset operation
    auto result = storage_->multiset(d_keys, d_values, batch_size);
    

    if (result != mlkv_plus::OperationResult::SUCCESS) {
        throw std::runtime_error("MLKV Plus multiset operation failed");
    }

}

template<typename K, typename V>
void MLKVPlusBinding<K, V>::multiset_for_loading(uint32_t batch_size,
                                    const K* d_keys,
                                    const V* d_values,
                                    const CallContext& ctx) {
    if (!storage_) {
        throw std::runtime_error("MLKV Plus binding not initialized");
    }

    // Perform multiset operation
    auto result = storage_->multiset(d_keys, d_values, batch_size);
    

    if (result != mlkv_plus::OperationResult::SUCCESS) {
        throw std::runtime_error("MLKV Plus multiset operation failed");
    }

}

template<typename K, typename V>
void MLKVPlusBinding<K, V>::multiget(uint32_t batch_size,
                                    const K* d_keys,
                                    V* d_values_out,
                                    bool* d_found,
                                    const CallContext& ctx) {
    

    auto result = storage_->multiget(d_keys, d_values_out, d_found, batch_size);

    if (result != mlkv_plus::OperationResult::SUCCESS) {
        throw std::runtime_error("MLKV Plus multiget operation failed");
    }

}


template<typename K, typename V>
void MLKVPlusBinding<K, V>::get(const K* h_keys,
                               V* d_values_out,
                               bool* h_found,
                               const CallContext& ctx) {

    K* d_keys;
    MLKV_CUDA_CHECK(cudaMalloc(&d_keys, sizeof(K)));
    MLKV_CUDA_CHECK(cudaMemcpy(d_keys, h_keys, sizeof(K), cudaMemcpyHostToDevice));

    auto result = storage_->get(d_keys, d_values_out);


    if (result == mlkv_plus::OperationResult::KEY_NOT_FOUND) {
        memset(h_found, false,  sizeof(bool));
    } else {
        memset(h_found, true,  sizeof(bool));
    }
    
    cudaFree(d_keys);

}



using MLKVPlusBinding_u64d = MLKVPlusBinding<uint64_t, double>;
REGISTER_CUDA_BINDING(uint64_t, double, MLKVPlusBinding_u64d, "mlkv_plus");