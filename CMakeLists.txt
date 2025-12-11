cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(SET CMP0077 NEW)
project(mlkv_plus LANGUAGES CUDA CXX)
# ------------------------------------------------------------
# Export compile commands for clangd
set(CMAKE_CUDA_USE_RESPONSE_FILE OFF CACHE BOOL "" FORCE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF CACHE BOOL "" FORCE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS  OFF CACHE BOOL "" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()



find_package(CUDAToolkit REQUIRED)


foreach(cuda_arch ${sm})
  list(APPEND cuda_arch_list ${cuda_arch})
  message(STATUS "Assign GPU architecture (sm=${cuda_arch})")
endforeach()

list(LENGTH cuda_arch_list cuda_arch_list_length)
if(cuda_arch_list_length EQUAL 0)
  list(APPEND cuda_arch_list "86")
  message(STATUS "Assign default GPU architecture sm=86")
endif()

foreach(cuda_arch ${cuda_arch_list})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${cuda_arch},code=sm_${cuda_arch}")
endforeach()

find_package(OpenMP REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)



add_subdirectory(libgycsb)


add_subdirectory(bindings_cpp)
