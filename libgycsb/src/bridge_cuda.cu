#include "bridge_cuda.cuh"
#include "binding_registry.h"
#include "benchmark_util.h"
#include <functional>
#include <set>


using benchmark::Timer;


template<typename K, typename V>
YCSBBridgeCUDA<K, V>::YCSBBridgeCUDA(const std::string& binding_name) {
    BindingInfo<K, V>* binding_info = BindingRegistry<K, V>::getInstance().getBindingInfo(binding_name);


    if (!binding_info->isCuda){
        std::cerr << binding_name + " is not CUDA binding." << std::endl;
        throw std::runtime_error(binding_name + " is not CUDA binding.");
    }

    binding_ = binding_info->factory();
}

template<typename K, typename V>
YCSBBridgeCUDA<K, V>::~YCSBBridgeCUDA() {
    std::cout << "YCSBBridgeCUDA destructor" << std::endl;
    
}


template<typename K, typename V>
void YCSBBridgeCUDA<K, V>::initialize(uint64_t gpu_init_capacity, uint64_t gpu_max_capacity, uint32_t dim, uint32_t hbm_gb, std::vector<int> gpu_ids, uint64_t max_batch_size, const std::string& binding_config) {
    InitConfig cfg;
    dim_ = dim;
    hbm_gb_ = hbm_gb;
    gpu_ids_ = gpu_ids;
    max_batch_size_ = max_batch_size;
    binding_config_ = binding_config;

    cfg.dim = dim_;
    cfg.max_batch_size = max_batch_size_;
    cfg.additional_config = binding_config_;
    cfg.use_cuda = true;
    cfg.init_capacity = gpu_init_capacity;
    cfg.max_capacity = gpu_max_capacity;
    cfg.hbm_gb = hbm_gb_;
    cfg.gpu_ids = gpu_ids_;

    bool init_success = binding_->initialize(cfg);
    if (!init_success) {
        std::cerr << "Failed to initialize binding" << std::endl;
        throw std::runtime_error("Failed to initialize binding");
    }
}

template<typename K, typename V>
void YCSBBridgeCUDA<K, V>::multiset_for_loading(uint32_t batch_size, const K* keys, const V* values, cudaStream_t stream) {
    // Need to transfer data to device first
    K* d_keys;
    V* d_values;
    cudaMalloc(&d_keys, batch_size * sizeof(K));
    cudaMalloc(&d_values, batch_size * dim_ * sizeof(V));
    cudaMemcpy(d_keys, keys, batch_size * sizeof(K), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, batch_size * dim_ * sizeof(V), cudaMemcpyHostToDevice);
    
    binding_->multiset_for_loading(batch_size, d_keys, d_values, {reinterpret_cast<std::uintptr_t>(stream)});
    
    cudaFree(d_keys);
    cudaFree(d_values);
}

template<typename K, typename V>
BenchmarkResult YCSBBridgeCUDA<K, V>::run_benchmark(const std::vector<Operation<K, V>>& ops, uint64_t num_streams, bool data_integrity) {

    if (gpu_ids_.size() == 1) {
        cudaSetDevice(gpu_ids_[0]);
    }
    
    
    // Create a vector of CUDA streams
    std::vector<cudaStream_t> streams;
    if (num_streams > 0) {
        streams.reserve(num_streams);
        for (uint32_t i = 0; i < num_streams; ++i) {
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            streams.push_back(stream);
        }
    }else {
        std::cout << "num_streams is 0, using default stream" << std::endl;
        streams.push_back(0);
        num_streams = 1;
    }

    // Prepare device memory for all operations (outside timing)
    std::vector<K*> d_keys_multiset_list;
    std::vector<V*> d_values_multiset_list;
    std::vector<K*> d_keys_read_list;
    std::vector<V*> d_values_out_list;
    std::vector<bool*> d_found_list;
    
    std::vector<std::function<void()>> workload_batch_fns;
    workload_batch_fns.reserve(ops.size());

    int read_counter = 0;
    int write_counter = 0;
    
    std::cout << "Preparing data transfer to GPU..." << std::endl;

    for (uint32_t i = 0; i < ops.size(); ++i) {

        uint32_t stream_idx = i % num_streams;

        if (ops[i].op == "multiget") {
            // Allocate device memory for keys, values output, and found flags
            K* d_keys;
            V* d_values_out;
            bool* d_found;

            cudaError_t err = cudaMalloc(&d_keys, ops[i].keys.size() * sizeof(K));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed");
            }
            
            err = cudaMalloc(&d_values_out, ops[i].keys.size() * dim_ * sizeof(V));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed");
            }
            
            err = cudaMalloc(&d_found, ops[i].keys.size() * sizeof(bool));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed");
            }

            cudaMemset(d_found, true, ops[i].keys.size() * sizeof(bool));
             
            // Transfer keys to device (outside timing)
            cudaMemcpy(d_keys, ops[i].keys.data(), ops[i].keys.size() * sizeof(K), cudaMemcpyHostToDevice);

            d_keys_read_list.push_back(d_keys);
            d_values_out_list.push_back(d_values_out);
            d_found_list.push_back(d_found);

            workload_batch_fns.push_back([&, i, read_counter, stream_idx]() {
                binding_->multiget(ops[i].keys.size(), d_keys_read_list[read_counter], d_values_out_list[read_counter], d_found_list[read_counter], {reinterpret_cast<std::uintptr_t>(streams[stream_idx])});
            });
            read_counter++;
            
        } else if (ops[i].op == "multiset") {
            // Allocate device memory for keys and values
            K* d_keys;
            V* d_values;
            
            cudaError_t err = cudaMalloc(&d_keys, ops[i].keys.size() * sizeof(K));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed");
            }
            
            err = cudaMalloc(&d_values, ops[i].keys.size() * dim_ * sizeof(V));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed");
            }

            // Transfer keys and values to device (outside timing)
            cudaMemcpy(d_keys, ops[i].keys.data(), ops[i].keys.size() * sizeof(K), cudaMemcpyHostToDevice);
            cudaMemcpy(d_values, ops[i].values.data(), ops[i].keys.size() * dim_ * sizeof(V), cudaMemcpyHostToDevice);

            d_keys_multiset_list.push_back(d_keys);
            d_values_multiset_list.push_back(d_values);

            workload_batch_fns.push_back([&, i, write_counter, stream_idx]() {
                binding_->multiset(ops[i].keys.size(), d_keys_multiset_list[write_counter], d_values_multiset_list[write_counter], {reinterpret_cast<std::uintptr_t>(streams[stream_idx])});
            });
            write_counter++;
            
        } else if (ops[i].op == "multiput") {
            // multiput is the same as multiset but uses existing keys
            // Allocate device memory for keys and values
            K* d_keys;
            V* d_values;
            
            cudaError_t err = cudaMalloc(&d_keys, ops[i].keys.size() * sizeof(K));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed");
            }
            
            err = cudaMalloc(&d_values, ops[i].keys.size() * dim_ * sizeof(V));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed");
            }

            // Transfer keys and values to device (outside timing)
            cudaMemcpy(d_keys, ops[i].keys.data(), ops[i].keys.size() * sizeof(K), cudaMemcpyHostToDevice);
            cudaMemcpy(d_values, ops[i].values.data(), ops[i].keys.size() * dim_ * sizeof(V), cudaMemcpyHostToDevice);

            d_keys_multiset_list.push_back(d_keys);
            d_values_multiset_list.push_back(d_values);

            workload_batch_fns.push_back([&, i, write_counter, stream_idx]() {
                binding_->multiput(ops[i].keys.size(), d_keys_multiset_list[write_counter], d_values_multiset_list[write_counter], {reinterpret_cast<std::uintptr_t>(streams[stream_idx])});
            });
            write_counter++;
            
        } else if (ops[i].op == "read") {
            K* d_keys;
            V* d_values_out;
            bool* d_found;

            cudaError_t err = cudaMalloc(&d_keys, sizeof(K));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed");
            }
            
            err = cudaMalloc(&d_values_out, dim_ * sizeof(V));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed");
            }
            
            err = cudaMalloc(&d_found, sizeof(bool));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed");
            }

            // Transfer keys to device (outside timing)
            cudaMemcpy(d_keys, ops[i].keys.data(), sizeof(K), cudaMemcpyHostToDevice);

            d_keys_read_list.push_back(d_keys);
            d_values_out_list.push_back(d_values_out);
            d_found_list.push_back(d_found);
            
            workload_batch_fns.push_back([&, i, read_counter, stream_idx]() {
                binding_->get(d_keys_read_list[read_counter], d_values_out_list[read_counter], d_found_list[read_counter], {reinterpret_cast<std::uintptr_t>(streams[stream_idx])});
            });
            read_counter++;
        }
        else {
            std::cerr << "Unsupported operation: " << ops[i].op << std::endl;
            throw std::runtime_error("Unsupported operation");
        }
    }
    
    cudaDeviceSynchronize(); // Ensure all data is transferred before timing
    std::cout << "Data transfer complete" << std::endl;

    std::cout << "start running workload..." << std::endl;

    // ===============================
    // run workload
    // ===============================

    Timer<double> timer;
    timer.start();
    for (const auto& fn : workload_batch_fns) {
        fn();
    }
    cudaDeviceSynchronize();
    timer.end();
    double total_time = timer.getResult();
    std::cout << "workload done" << std::endl;


    // ===============================
    // verify integrity
    // ===============================
    std::pair<bool, double> result;
    if (data_integrity) {
        result = verify_integrity_ycsb(ops, d_values_out_list, d_found_list);
    } else {
        std::cout << "Data integrity is not checked" << std::endl;
        result = std::make_pair(true, 1.0);
    }

    // Clean up device memory
    for (auto& d_keys : d_keys_multiset_list) {
        cudaFree(d_keys);
    }
    for (auto& d_values : d_values_multiset_list) {
        cudaFree(d_values);
    }
    for (auto& d_keys : d_keys_read_list) {
        cudaFree(d_keys);
    }
    for (auto& d_values_out : d_values_out_list) {
        cudaFree(d_values_out);
    }
    for (auto& d_found : d_found_list) {
        cudaFree(d_found);
    }
    
    // Clean up streams
    for (auto& stream : streams) {
        if (stream != 0) {
            cudaStreamDestroy(stream);
        }
    }

    return BenchmarkResult{total_time, result.first, result.second};
}

// GPU kernel to verify the correctness of retrieved values
template<typename K, typename V>
__global__ void verify_values_kernel(const K* keys, const V* values, bool* found, 
                                    uint32_t batch_size, uint32_t dim, bool* verification_results) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size && found[idx]) {
        // Generate expected value using the same logic as build_deterministic_value
        // For double type: (key + fieldkeyIndex) % 2^53
        // fieldkeyIndex is the dimension index (0 to dim-1)
        
        bool all_correct = true;
        for (uint32_t field_idx = 0; field_idx < dim; ++field_idx) {
            V expected_value;
            if constexpr (std::is_same_v<V, double>) {
                expected_value = static_cast<V>((keys[idx] + field_idx) % (1ULL << 53));
            } else if constexpr (std::is_same_v<V, float>) {
                expected_value = static_cast<V>((keys[idx] + field_idx) % (1ULL << 53));
            } else {
                expected_value = static_cast<V>((keys[idx] + field_idx) % (1ULL << 53));
            }
            
            V actual_value = values[idx * dim + field_idx];
            V diff = actual_value - expected_value;
            if constexpr (std::is_floating_point_v<V>) {
                if (fabs(diff) > 1e-5) {
                    all_correct = false;
                    break;
                }
            } else {
                if (diff != 0) {
                    all_correct = false;
                    break;
                }
            }
        }
        verification_results[idx] = all_correct;
    } else {
        verification_results[idx] = true; // Not found keys are considered correct for this check
    }
}

template<typename K, typename V>
std::pair<bool, double> YCSBBridgeCUDA<K, V>::verify_integrity_ycsb(const std::vector<Operation<K, V>>& stored_ops, const std::vector<V*>& d_values_out_list, const std::vector<bool*>& d_found_list) {


    if (gpu_ids_.size() == 1) {
        cudaSetDevice(gpu_ids_[0]);
    } else {
        std::cerr << "Data integrity check is not supported for multiple GPUs" << std::endl;
        throw std::runtime_error("Data integrity check is not supported for multiple GPUs");
    }
    
    bool overall_result = true;
    size_t read_op_idx = 0;
    double overall_accurate_count = 0;
    double overall_found_count = 0;
    double overall_op_count = 0;

    std::cout << "Starting integrity verification..." << std::endl;
    
    // First pass: verify multiget and read operations
    for (size_t op_idx = 0; op_idx < stored_ops.size(); ++op_idx) {
        if (stored_ops[op_idx].op == "multiget" || stored_ops[op_idx].op == "read") {
            V* d_values_out = d_values_out_list[read_op_idx];
            bool* d_found = d_found_list[read_op_idx];
            uint32_t batch_size = stored_ops[op_idx].keys.size();
            overall_op_count += batch_size;
            
            K* d_keys;
            bool* d_verification_results;
            
            cudaMalloc(&d_keys, batch_size * sizeof(K));
            cudaMalloc(&d_verification_results, batch_size * sizeof(bool));
            
            // Copy keys to device
            cudaMemcpy(d_keys, stored_ops[op_idx].keys.data(), 
                        batch_size * sizeof(K), cudaMemcpyHostToDevice);
            
            // Launch verification kernel
            int blockSize = 256;
            int numBlocks = (batch_size + blockSize - 1) / blockSize;
            
            verify_values_kernel<<<numBlocks, blockSize>>>(
                d_keys, d_values_out, d_found, batch_size, dim_, d_verification_results);
            
            cudaDeviceSynchronize();
            
            // Copy verification results and found flags back to host
            bool* h_verification_results = new bool[batch_size];
            bool* h_found = new bool[batch_size];
            cudaMemcpy(h_verification_results, d_verification_results, 
                        batch_size * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_found, d_found, batch_size * sizeof(bool), cudaMemcpyDeviceToHost);
            
            // Check results
            int found_count = 0;
            int correct_count = 0;
            for (uint32_t i = 0; i < batch_size; ++i) {
                if (h_found[i]) {
                    found_count++;
                    if (h_verification_results[i]) {
                        correct_count++;
                    } else {
                        // Copy values from device to host for printing
                        V* h_values = new V[dim_];
                        cudaMemcpy(h_values, d_values_out + i * dim_, dim_ * sizeof(V), cudaMemcpyDeviceToHost);
                        // std::cout << "Key " << stored_ops[op_idx].keys[i] 
                        //           << " Value[0]: " << h_values[0]
                        //           << " Value[1]: " << h_values[1]
                        //           << " Value[2]: " << h_values[2]
                        //           << " Value[3]: " << h_values[3]
                        //           << " Value[4]: " << h_values[4]
                        //           << " has incorrect value!" << std::endl;
                        delete[] h_values;
                        overall_result = false;
                    }
                } else {
                    // std::cout << "Key " << stored_ops[op_idx].keys[i] 
                    //           << " not found!" << std::endl;
                    overall_result = false;
                }
            }
            overall_accurate_count += correct_count;
            overall_found_count += found_count;
            
            // Cleanup memory
            cudaFree(d_keys);
            cudaFree(d_verification_results);
            delete[] h_verification_results;
            delete[] h_found;

            read_op_idx++;

        }
    }

    // Second pass: verify multiset and multiput operations
    // Collect all unique keys from multiset and multiput operations
    std::set<K> unique_keys_set;
    for (size_t op_idx = 0; op_idx < stored_ops.size(); ++op_idx) {
        if (stored_ops[op_idx].op == "multiset" || stored_ops[op_idx].op == "multiput") {
            for (const auto& key : stored_ops[op_idx].keys) {
                unique_keys_set.insert(key);
            }
        }
    }
    
    if (!unique_keys_set.empty()) {
        std::cout << "Verifying multiset/multiput operations, total unique keys: " << unique_keys_set.size() << std::endl;
        
        // Convert set to vector for easier batching
        std::vector<K> unique_keys(unique_keys_set.begin(), unique_keys_set.end());
        size_t total_keys = unique_keys.size();
        overall_op_count += total_keys;
        
        // Process keys in batches according to max_batch_size_
        for (size_t key_offset = 0; key_offset < total_keys; key_offset += max_batch_size_) {
            uint32_t batch_size = std::min(static_cast<size_t>(max_batch_size_), total_keys - key_offset);
            
            // Allocate device memory for this batch
            K* d_keys;
            V* d_values_out;
            bool* d_found;
            
            cudaError_t err = cudaMalloc(&d_keys, batch_size * sizeof(K));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed for multiset verification: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed for multiset verification");
            }
            
            err = cudaMalloc(&d_values_out, batch_size * dim_ * sizeof(V));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed for multiset verification: " << cudaGetErrorString(err) << std::endl;
                cudaFree(d_keys);
                throw std::runtime_error("cudaMalloc failed for multiset verification");
            }
            
            err = cudaMalloc(&d_found, batch_size * sizeof(bool));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed for multiset verification: " << cudaGetErrorString(err) << std::endl;
                cudaFree(d_keys);
                cudaFree(d_values_out);
                throw std::runtime_error("cudaMalloc failed for multiset verification");
            }
            
            // Copy keys to device
            cudaMemcpy(d_keys, unique_keys.data() + key_offset, 
                        batch_size * sizeof(K), cudaMemcpyHostToDevice);
            
            // Create a stream for this operation
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            
            // Call multiget on the binding to retrieve values
            binding_->multiget(batch_size, d_keys, d_values_out, d_found, {reinterpret_cast<std::uintptr_t>(stream)});
            cudaStreamSynchronize(stream);
            
            // Verify the retrieved values using the kernel
            bool* d_verification_results;
            cudaMalloc(&d_verification_results, batch_size * sizeof(bool));
            
            // Launch verification kernel
            int blockSize = 256;
            int numBlocks = (batch_size + blockSize - 1) / blockSize;
            
            verify_values_kernel<<<numBlocks, blockSize>>>(
                d_keys, d_values_out, d_found, batch_size, dim_, d_verification_results);
            
            cudaDeviceSynchronize();
            
            // Copy verification results and found flags back to host
            bool* h_verification_results = new bool[batch_size];
            bool* h_found = new bool[batch_size];
            cudaMemcpy(h_verification_results, d_verification_results, 
                        batch_size * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_found, d_found, batch_size * sizeof(bool), cudaMemcpyDeviceToHost);
            
            // Check results
            int found_count = 0;
            int correct_count = 0;
            for (uint32_t i = 0; i < batch_size; ++i) {
                if (h_found[i]) {
                    found_count++;
                    if (h_verification_results[i]) {
                        correct_count++;
                    } else {
                        // std::cout << "Multiset key " << unique_keys[key_offset + i] 
                        //             << " has incorrect value!" << std::endl;
                        overall_result = false;
                    }
                } else {
                    // std::cout << "Multiset key " << unique_keys[key_offset + i] 
                    //           << " not found!" << std::endl;
                    overall_result = false;
                }
            }
            overall_accurate_count += correct_count;
            overall_found_count += found_count;
            
            // Cleanup
            cudaFree(d_keys);
            cudaFree(d_verification_results);
            cudaFree(d_found);
            cudaFree(d_values_out);
            cudaStreamDestroy(stream);
            delete[] h_found;
            delete[] h_verification_results;
        }
    }

    std::cout << "Found rate: " << overall_found_count / overall_op_count << std::endl;
    std::cout << "Accurate rate in found keys: " << overall_accurate_count / overall_found_count << std::endl;
    std::cout << "Overall accurate rate: " << overall_accurate_count / overall_op_count << std::endl;
    return std::make_pair(overall_result, overall_accurate_count / overall_op_count);
}

template<typename K, typename V>
void YCSBBridgeCUDA<K, V>::cleanup() {
    binding_->cleanup();
}


template<typename K, typename V>
std::vector<std::string> YCSBBridgeCUDA<K, V>::getAvailableBindings() {
    return BindingRegistry<K, V>::getInstance().getAvailableBindings(true);
}




// Explicit template instantiations for supported types
template class YCSBBridgeCUDA<uint64_t, double>;
template class YCSBBridgeCUDA<uint64_t, float>;