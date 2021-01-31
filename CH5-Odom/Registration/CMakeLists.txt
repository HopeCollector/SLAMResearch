cmake_minimum_required(VERSION 3.10)

project(Registration
            VERSION 0.1
            LANGUAGES CXX)

find_package(PCL REQUIRED COMPONENTS
    common 
    io 
    registration 
    visualization 
    features 
    filters)
find_package(Boost REQUIRED COMPONENTS 
    program_options)

add_executable(icp)
target_sources(icp PRIVATE icp.cpp)
target_compile_features(icp PUBLIC cxx_std_17)
target_link_libraries(icp PRIVATE
    ${PCL_LIBRARIES}
    Boost::program_options
)

add_executable(icp_incremental)
target_sources(icp_incremental PRIVATE icp_incremental.cpp)
target_compile_features(icp_incremental PUBLIC cxx_std_17)
target_link_libraries(icp_incremental PRIVATE
    ${PCL_LIBRARIES}
    Boost::program_options
)

add_executable(ndt)
target_sources(ndt PRIVATE ndt.cpp)
target_compile_features(ndt PUBLIC cxx_std_17)
target_link_libraries(ndt PRIVATE
    ${PCL_LIBRARIES}
    Boost::program_options
)