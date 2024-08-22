cmake_minimum_required(VERSION 3.5)

project(Measure_size LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package( OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



add_executable(Measure_size
        my_minarearect.cpp
        myconvexhull.cpp
        mycontour.cpp
        measure_size.cpp
)
target_link_libraries(Measure_size ${OpenCV_LIBS})
