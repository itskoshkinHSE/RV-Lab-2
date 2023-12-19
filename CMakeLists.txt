cmake_minimum_required(VERSION 3.27)
project(RV_Lab_2 C)
set(CMAKE_C_STANDARD 23)

add_executable(RV_Lab_2 main-columns.c)
#add_executable(RV_Lab_2 main-strings.c)
#add_executable(RV_Lab_2 main-blocks.c)
#add_executable(RV_Lab_2 main.c)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(RV_Lab_2 ${MPI_C_LIBRARIES})
