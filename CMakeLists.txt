cmake_minimum_required(VERSION 3.16)
project(scc-spbstu-master2025 C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

enable_testing()
add_subdirectory(common)
add_subdirectory(single_thread)
add_subdirectory(tests)