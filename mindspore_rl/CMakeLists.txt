cmake_minimum_required(VERSION 3.14.0)
project(MindSpore_RL)

function(redefine_file_macro targetname)
    get_target_property(source_files "${targetname}" SOURCES)
    foreach(sourcefile ${source_files})
        get_property(defs SOURCE "${sourcefile}"
            PROPERTY COMPILE_DEFINITIONS)
        get_filename_component(filepath "${sourcefile}" ABSOLUTE)
        string(REPLACE ${PROJECT_SOURCE_DIR}/ "" relpath ${filepath})
        list(APPEND defs "__FILE__=\"${relpath}\"")
        set_property(
            SOURCE "${sourcefile}"
            PROPERTY COMPILE_DEFINITIONS ${defs}
        )
    endforeach()
endfunction()

if(DEBUG_MODE)
    set(CMAKE_BUILD_TYPE "DEBUG")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC")

file(GLOB MCTS_COMMON_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "utils/mcts/*.cc")

file(GLOB_RECURSE MCTS_SRC_CPU RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "utils/mcts/cpu/*.cc")
list(APPEND MCTS_SRC_CPU ${MCTS_COMMON_SRC})

set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
add_library(mcts_cpu SHARED ${MCTS_SRC_CPU})
install(TARGETS mcts_cpu DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/mindspore_rl/utils/mcts")
redefine_file_macro(mcts_cpu)

if(ENABLE_GPU)
    find_package(CUDA REQUIRED)
    include_directories((${CUDA_INCLUDE_DIRS}))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_53 --expt-relaxed-constexpr")

    if(DEBUG_MODE)
        list(APPEND CUDA_NVCC_FLAGS "-g -G")
    endif()

    file(GLOB_RECURSE MCTS_SRC_GPU RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "utils/mcts/gpu/*.cc" "utils/mcts/gpu/*.cu")
    list(APPEND MCTS_SRC_GPU ${MCTS_COMMON_SRC})
    cuda_add_library(mcts_gpu SHARED ${MCTS_SRC_GPU})
    install(TARGETS mcts_gpu DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/mindspore_rl/utils/mcts")
    redefine_file_macro(mcts_gpu)
endif()
