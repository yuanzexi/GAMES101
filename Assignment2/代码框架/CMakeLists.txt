cmake_minimum_required(VERSION 3.10)
project(Rasterizer)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

# include_directories(/usr/local/include)

if (EIGEN_ROOT)
  include_directories(${EIGEN_ROOT})
else()
  message(FATAL_ERROR "EIGEN_ROOT NOT-FOUND")
endif()

add_executable(Rasterizer main.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp)
target_link_libraries(Rasterizer ${OpenCV_LIBRARIES})
