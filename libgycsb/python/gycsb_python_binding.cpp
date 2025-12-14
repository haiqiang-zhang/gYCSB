#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "bridge.h"
#ifdef HAVE_CUDA
#include "bridge_cuda.cuh"
#endif
#include <string>

namespace py = pybind11;

// Macro to bind Operation for a specific value type
#define BIND_OPERATION(VALUE_TYPE, VALUE_TYPE_NAME) \
    py::class_<Operation<uint64_t, VALUE_TYPE>>(m, "Operation_uint64_" VALUE_TYPE_NAME) \
        .def(py::init<const std::string&, const std::vector<uint64_t>&, const std::vector<VALUE_TYPE>&>()) \
        .def_readwrite("op", &Operation<uint64_t, VALUE_TYPE>::op) \
        .def_readwrite("keys", &Operation<uint64_t, VALUE_TYPE>::keys) \
        .def_readwrite("values", &Operation<uint64_t, VALUE_TYPE>::values);

// Macro to bind YCSBBridgeCUDA for a specific value type (only if CUDA is available)
#ifdef HAVE_CUDA
#define BIND_YCSB_CUDA(VALUE_TYPE, VALUE_TYPE_NAME) \
    py::class_<YCSBBridgeCUDA<uint64_t, VALUE_TYPE>>(m, "YCSBBridgeCUDA_uint64_" VALUE_TYPE_NAME) \
        .def(py::init<const std::string&>(), py::arg("binding_name")) \
        .def("initialize", &YCSBBridgeCUDA<uint64_t, VALUE_TYPE>::initialize, \
             "Initialize the bridge with specified parameters", \
             py::arg("gpu_init_capacity"), py::arg("gpu_max_capacity"), py::arg("dim"), py::arg("hbm_gb"), \
             py::arg("gpu_id"), py::arg("max_batch_size"), py::arg("binding_config")) \
        .def("multiset_for_loading", [](YCSBBridgeCUDA<uint64_t, VALUE_TYPE>& self, \
                           uint32_t batch_size, \
                           py::array_t<uint64_t> keys, \
                           py::array_t<VALUE_TYPE> values, \
                           py::object stream) { \
            if (keys.size() != batch_size) { \
                throw std::runtime_error("Keys array size (" + std::to_string(keys.size()) + \
                                       ") does not match batch_size (" + std::to_string(batch_size) + ")"); \
            } \
            const uint64_t* keys_ptr = static_cast<const uint64_t*>(keys.data()); \
            const VALUE_TYPE* values_ptr = static_cast<const VALUE_TYPE*>(values.data()); \
            cudaStream_t stream_ptr = 0; \
            if (!stream.is_none()) { \
                stream_ptr = reinterpret_cast<cudaStream_t>(stream.cast<std::uintptr_t>()); \
            } \
            self.multiset_for_loading(batch_size, keys_ptr, values_ptr, stream_ptr); \
        }, "Perform a multiset operation", \
           py::arg("batch_size"), py::arg("keys"), py::arg("values"), py::arg("stream") = py::none()) \
        .def("run_benchmark", [](YCSBBridgeCUDA<uint64_t, VALUE_TYPE>& self, \
                                 std::vector<Operation<uint64_t, VALUE_TYPE>>& ops, \
                                 uint64_t num_streams, \
                                 bool data_integrity) { \
            return self.run_benchmark(ops, num_streams, data_integrity); \
        }, \
             "Run the benchmark", \
             py::arg("ops"), py::arg("num_streams") = 1, py::arg("data_integrity") = true) \
        .def("cleanup", &YCSBBridgeCUDA<uint64_t, VALUE_TYPE>::cleanup, \
             "Cleanup the bridge") \
        .def_static("get_available_bindings", &YCSBBridgeCUDA<uint64_t, VALUE_TYPE>::getAvailableBindings, \
                   "Get list of available bindings for uint64_t keys and " VALUE_TYPE_NAME " values");
#else
#define BIND_YCSB_CUDA(VALUE_TYPE, VALUE_TYPE_NAME) /* CUDA not available */
#endif

// Macro to bind YCSBBridgeCPU for a specific value type
#define BIND_YCSB_CPU(VALUE_TYPE, VALUE_TYPE_NAME) \
    py::class_<YCSBBridgeCPU<uint64_t, VALUE_TYPE>>(m, "YCSBBridgeCPU_uint64_" VALUE_TYPE_NAME) \
        .def(py::init<const std::string&>(), py::arg("binding_name")) \
        .def("initialize", &YCSBBridgeCPU<uint64_t, VALUE_TYPE>::initialize, \
             "Initialize the CPU bridge with specified parameters", \
             py::arg("dim"), py::arg("max_batch_size"), py::arg("binding_config")) \
        .def("multiset_for_loading", [](YCSBBridgeCPU<uint64_t, VALUE_TYPE>& self, \
                           uint32_t batch_size, \
                           py::array_t<uint64_t> keys, \
                           py::array_t<VALUE_TYPE> values) { \
            if (keys.size() != batch_size) { \
                throw std::runtime_error("Keys array size (" + std::to_string(keys.size()) + \
                                       ") does not match batch_size (" + std::to_string(batch_size) + ")"); \
            } \
            const uint64_t* keys_ptr = static_cast<const uint64_t*>(keys.data()); \
            const VALUE_TYPE* values_ptr = static_cast<const VALUE_TYPE*>(values.data()); \
            self.multiset_for_loading(batch_size, keys_ptr, values_ptr); \
        }, "Perform a multiset operation on CPU", \
           py::arg("batch_size"), py::arg("keys"), py::arg("values")) \
        .def("run_benchmark", [](YCSBBridgeCPU<uint64_t, VALUE_TYPE>& self, \
                                 std::vector<Operation<uint64_t, VALUE_TYPE>>& ops, \
                                 bool data_integrity) { \
            return self.run_benchmark(ops, data_integrity); \
        }, \
             "Run the benchmark on CPU", \
             py::arg("ops"), py::arg("data_integrity") = true) \
        .def("cleanup", &YCSBBridgeCPU<uint64_t, VALUE_TYPE>::cleanup, \
             "Cleanup the CPU bridge") \
        .def_static("get_available_bindings", &YCSBBridgeCPU<uint64_t, VALUE_TYPE>::getAvailableBindings, \
                   "Get list of available CPU bindings for uint64_t keys and " VALUE_TYPE_NAME " values");


PYBIND11_MODULE(gycsb_python_binding, m) {
    m.doc() = "YCSB Benchmark Python Bindings";

    // Generate bindings for double
    BIND_OPERATION(double, "double")
    BIND_YCSB_CUDA(double, "double")
    BIND_YCSB_CPU(double, "double")
    
    // Generate bindings for float
    BIND_OPERATION(float, "float")
    BIND_YCSB_CUDA(float, "float")
    BIND_YCSB_CPU(float, "float")

    // Expose BenchmarkResult struct
    py::class_<BenchmarkResult>(m, "BenchmarkResult")
        .def(py::init<>())
        .def_readwrite("time_seconds", &BenchmarkResult::time_seconds)
        .def_readwrite("integrity", &BenchmarkResult::integrity)
        .def_readwrite("integrity_accuracy", &BenchmarkResult::integrity_accuracy);
} 