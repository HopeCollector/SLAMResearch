cmake_minimum_required(VERSION 3.5)

project(transform)

execute_process(COMMAND "lsb_release" -rs
    OUTPUT_VARIABLE SYSTEM_RELEASE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(SYSTEM_RELEASE MATCHES "20.04")
  message("Using c++14")
  add_compile_options(-std=c++14)
else()
  message("Using c++11")
  add_compile_options(-std=c++11)
endif()

find_package(Eigen3 REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC
    Eigen3::Eigen
)