# ------------------
# Function to link binding libraries to bridge_cpu or bridge_cuda
# Usage: link_binding_to_bridge(BRIDGE_TYPE CPU|CUDA binding_target1 [binding_target2 ...])
# This function automatically links bindings to bridge
# - If bridge already exists: links immediately
# - If bridge doesn't exist yet: registers for later linking
# ------------------
function(link_binding_to_bridge)
    set(options)
    set(oneValueArgs BRIDGE_TYPE)
    set(multiValueArgs)
    cmake_parse_arguments(LINK_BINDING "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    if(NOT LINK_BINDING_BRIDGE_TYPE)
        message(FATAL_ERROR "link_binding_to_bridge: BRIDGE_TYPE must be specified (CPU or CUDA)")
    endif()
    
    string(TOUPPER "${LINK_BINDING_BRIDGE_TYPE}" bridge_type_upper)
    
    if(bridge_type_upper STREQUAL "CPU")
        # Check if bridge_cpu already exists
        if(TARGET bridge_cpu)
            # Bridge exists, link immediately
            foreach(binding_target ${LINK_BINDING_UNPARSED_ARGUMENTS})
                if(TARGET ${binding_target})
                    target_link_libraries(bridge_cpu PRIVATE -Wl,--no-as-needed ${binding_target})
                    message(STATUS "Linked ${binding_target} to bridge_cpu")
                else()
                    message(WARNING "Binding target ${binding_target} not found, skipping.")
                endif()
            endforeach()
        else()
            # Bridge doesn't exist yet, register for later
            if(DEFINED BRIDGE_CPU_BINDINGS)
                set(current_bindings ${BRIDGE_CPU_BINDINGS})
            else()
                set(current_bindings "")
            endif()
            foreach(binding_target ${LINK_BINDING_UNPARSED_ARGUMENTS})
                list(APPEND current_bindings ${binding_target})
            endforeach()
            set(BRIDGE_CPU_BINDINGS ${current_bindings} CACHE INTERNAL "List of bindings to link to bridge_cpu" FORCE)
            message(STATUS "Registered bindings to be linked to bridge_cpu: ${LINK_BINDING_UNPARSED_ARGUMENTS}")
        endif()
    elseif(bridge_type_upper STREQUAL "CUDA")
        if(NOT HAVE_CUDA)
            message(FATAL_ERROR "link_binding_to_bridge: CUDA is not available. Cannot link bindings to bridge_cuda.")
        endif()
        # Check if bridge_cuda already exists
        if(TARGET bridge_cuda)
            # Bridge exists, link immediately
            foreach(binding_target ${LINK_BINDING_UNPARSED_ARGUMENTS})
                if(TARGET ${binding_target})
                    target_link_libraries(bridge_cuda PRIVATE -Wl,--no-as-needed ${binding_target})
                    message(STATUS "Linked ${binding_target} to bridge_cuda")
                else()
                    message(WARNING "Binding target ${binding_target} not found, skipping.")
                endif()
            endforeach()
        else()
            # Bridge doesn't exist yet, register for later
            if(DEFINED BRIDGE_CUDA_BINDINGS)
                set(current_bindings ${BRIDGE_CUDA_BINDINGS})
            else()
                set(current_bindings "")
            endif()
            foreach(binding_target ${LINK_BINDING_UNPARSED_ARGUMENTS})
                list(APPEND current_bindings ${binding_target})
            endforeach()
            set(BRIDGE_CUDA_BINDINGS ${current_bindings} CACHE INTERNAL "List of bindings to link to bridge_cuda" FORCE)
            message(STATUS "Registered bindings to be linked to bridge_cuda: ${LINK_BINDING_UNPARSED_ARGUMENTS}")
        endif()
    else()
        message(FATAL_ERROR "link_binding_to_bridge: BRIDGE_TYPE must be either CPU or CUDA, got ${LINK_BINDING_BRIDGE_TYPE}")
    endif()
endfunction()

add_library(bridge_cpu SHARED src/bridge_cpu.cc)
target_compile_features(bridge_cpu PUBLIC cxx_std_20)
set_target_properties(bridge_cpu PROPERTIES 
    CUDA_ARCHITECTURES OFF 
    POSITION_INDEPENDENT_CODE ON
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH_USE_LINK_PATH ON
)
target_link_options(bridge_cpu PRIVATE "-Wl,--no-as-needed")

# ------------------------------------------------------------
# Please link the binding of CPU libraries to the bridge_cpu library
# Users should call link_binding_to_bridge() in their bindings_cpp/CMakeLists.txt
# to register their bindings with bridge_cpu
# ------------------------------------------------------------
# Process registered bindings
if(DEFINED BRIDGE_CPU_BINDINGS)
    foreach(binding_target ${BRIDGE_CPU_BINDINGS})
        if(TARGET ${binding_target})
            target_link_libraries(bridge_cpu PRIVATE -Wl,--no-as-needed ${binding_target})
            message(STATUS "Linked ${binding_target} to bridge_cpu")
        else()
            message(WARNING "Binding target ${binding_target} not found, skipping.")
        endif()
    endforeach()
endif()
# Example usage in bindings_cpp/CMakeLists.txt:
# link_binding_to_bridge(BRIDGE_TYPE CPU your_binding_name)
# ------------------------------------------------------------

# Only build bridge_cuda if CUDA is available
if(HAVE_CUDA)
add_library(bridge_cuda SHARED src/bridge_cuda.cu)
target_compile_features(bridge_cuda PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(bridge_cuda PROPERTIES 
    CUDA_ARCHITECTURES OFF 
    POSITION_INDEPENDENT_CODE ON
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH_USE_LINK_PATH ON
)

# ------------------------------------------------------------
# Please link the binding of CUDA libraries to the bridge_cuda library
# Users should call link_binding_to_bridge() in their bindings_cpp/CMakeLists.txt
# to register their bindings with bridge_cuda
# ------------------------------------------------------------
# Process registered bindings
if(DEFINED BRIDGE_CUDA_BINDINGS)
    foreach(binding_target ${BRIDGE_CUDA_BINDINGS})
        if(TARGET ${binding_target})
            target_link_libraries(bridge_cuda PRIVATE -Wl,--no-as-needed ${binding_target})
            message(STATUS "Linked ${binding_target} to bridge_cuda")
        else()
            message(WARNING "Binding target ${binding_target} not found, skipping.")
        endif()
    endforeach()
endif()
# Example usage in bindings_cpp/CMakeLists.txt:
# link_binding_to_bridge(BRIDGE_TYPE CUDA your_binding_name)
# ------------------------------------------------------------
endif()


# Add pybind11 subdirectory for Python bindings
add_subdirectory(python)
