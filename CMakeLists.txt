cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
project( eigenfaces )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( src )

set(HEADER_FILES src/file_functions.h src/eigenfaces.h src/set_definitions.h)
set(SOURCES src/main.cpp src/file_functions.cpp src/eigenfaces.cpp)

add_executable( eigenfaces ${SOURCES} ${HEAD_FILES})

target_link_libraries( eigenfaces ${OpenCV_LIBS} )