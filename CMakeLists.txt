cmake_minimum_required(VERSION 3.20)
project(tempo_dag LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only template library placeholder.
add_library(tempo_dag INTERFACE)
target_include_directories(tempo_dag INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
