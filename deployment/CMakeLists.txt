cmake_minimum_required(VERSION 3.16)

project(yatt_loader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(
    yatt_loader
    src/main.cpp
    src/model.cpp
    src/inference.cpp
    src/tokenizer.cpp
    src/generator.cpp
)

target_include_directories(
    yatt_loader
    PRIVATE
    include
)