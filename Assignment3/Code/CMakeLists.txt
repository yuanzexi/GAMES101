cmake_minimum_required(VERSION 3.10)
project(Rasterizer)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

if (EIGEN_ROOT)
  include_directories(${EIGEN_ROOT})
else()
  message(FATAL_ERROR "EIGEN_ROOT NOT-FOUND")
endif()

add_executable(Rasterizer main.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp Texture.hpp Texture.cpp Shader.hpp OBJ_Loader.h)
target_link_libraries(Rasterizer ${OpenCV_LIBRARIES})
#target_compile_options(Rasterizer PUBLIC -Wall -Wextra -pedantic)