# ------------------
# Add binding libraries (only if CUDA is available)
# ------------------
if(HAVE_CUDA)
add_library(hkv_binding SHARED ${CMAKE_CURRENT_SOURCE_DIR}/hkv_binding.cu)
set_target_properties(hkv_binding PROPERTIES 
    CUDA_ARCHITECTURES OFF 
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(hkv_binding PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(hkv_binding PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(hkv_binding PRIVATE ${PROJECT_SOURCE_DIR}/HierarchicalKV/include)
target_link_options(hkv_binding PRIVATE "-Wl,-Bsymbolic")
target_compile_options(hkv_binding PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link bindings to bridge_cuda using the helper function
# This function is defined in ycsb_cpp/CMakeLists.txt
# The function registers bindings to be linked when bridge_cuda is created
# This works regardless of the order of add_subdirectory calls
link_binding_to_bridge(BRIDGE_TYPE CUDA hkv_binding)



endif()