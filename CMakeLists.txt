cmake_minimum_required(VERSION 3.22.1)
project(MyProject VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 11)  # Set the C++ standard you are using

set(OpenCV_DIR "/home/pbhatia4/Desktop/opencv/build")
set(Sqlite3_DIR "/home/pbhatia4/Desktop/sqlite3")

include_directories(${OpenCV_INCLUDE_DIRS} ${Sqlite3_DIR} ${ADIOS2_INCLUDE_DIRS})

find_package(ADIOS2 REQUIRED)
find_package(OpenCV REQUIRED)

add_library(sqlite3_library STATIC ${Sqlite3_DIR}/sqlite3.c)
add_executable(executable executable.cpp)



target_link_libraries(executable PRIVATE sqlite3_library dl ${ADIOS2_LIBRARIES} ${OpenCV_LIBS} stdc++fs)
