# Find pybind11 - explicitly specify Python version to avoid conflicts
find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Debug: Print Python paths being used (can be removed in production)
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_VERSION: ${Python_VERSION}")


set(module_name gycsb_python_binding)

# Create the pybind11 module
pybind11_add_module(${module_name} gycsb_python_binding.cpp)

# Set properties
target_compile_features(${module_name} PRIVATE cxx_std_20)
set_target_properties(${module_name} PROPERTIES CUDA_ARCHITECTURES OFF)





# Get the relative path from project source to this submodule
# This handles cases where the submodule is named differently (e.g., "benchmark" instead of "gYCSB")
file(RELATIVE_PATH SUBMODULE_REL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.." "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Set RPATH to find the ycsb library and RocksDB
set(EXTRA_RPATH_DIRS
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_BINARY_DIR}/libgycsb"
    "${CMAKE_BINARY_DIR}/bindings_cpp"
    "${CMAKE_BINARY_DIR}/${SUBMODULE_REL_PATH}/libgycsb"
    "${CMAKE_BINARY_DIR}/${SUBMODULE_REL_PATH}/bindings_cpp"
)


set_target_properties(${module_name} PROPERTIES
    BUILD_RPATH "${EXTRA_RPATH_DIRS}"
    INSTALL_RPATH "${EXTRA_RPATH_DIRS}"
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH_USE_LINK_PATH ON)

# Link against the ycsb library and necessary dependencies
target_link_libraries(${module_name} PRIVATE bridge_cpu)
if(HAVE_CUDA)
    target_link_libraries(${module_name} PRIVATE bridge_cuda)
    target_compile_definitions(${module_name} PRIVATE HAVE_CUDA)
    # Add CUDA include directories so C++ compiler can find CUDA headers
    if(DEFINED CUDAToolkit_INCLUDE_DIRS)
        target_include_directories(${module_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    elseif(DEFINED CUDAToolkit_ROOT_DIR)
        target_include_directories(${module_name} PRIVATE ${CUDAToolkit_ROOT_DIR}/include)
    endif()
endif()


execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  OUTPUT_VARIABLE PYTHON_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "PYTHON_SITE: ${PYTHON_SITE}")

set(PYTHON_INSTALL_DIR "${PYTHON_SITE}/${module_name}")

# install the module to python site-packages
install(TARGETS ${module_name} 
        COMPONENT ${module_name}
        DESTINATION ${PYTHON_INSTALL_DIR})