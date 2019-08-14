cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(matching_algorithm)

find_package(OpenCV REQUIRED)
find_package(PCL 1.4 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (matching_algorithm matching_algorithm.h matching_algorithm.cpp)
target_link_libraries (matching_algorithm ${PCL_LIBRARIES})
target_link_libraries (matching_algorithm ${OpenCV_LIBS})


