cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(hello_trianlge)
add_executable(${PROJECT_NAME} main.cpp)

find_package(glfw3 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE glfw Vulkan::Vulkan)
