cmake_minimum_required(VERSION 3.14.0)
project(MindSpore_RL)

add_definitions(-Wno-builtin-macro-redefined)
set(RL_PATH ${CMAKE_SOURCE_DIR}/mindspore_rl)
include_directories(${RL_PATH})
add_subdirectory(${RL_PATH})