cmake_minimum_required(VERSION 2.8.7)
project(detect_circle)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fexceptions -frtti -pthread -O3 -march=core2")

find_package(OpenCV 2 REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

find_package(PCL 1.5 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


#添加头文件
set(ROOT 		"${CMAKE_CURRENT_SOURCE_DIR}/")
set(LIB_REALSENSE_INC "/usr/include")
include_directories(
	${LIB_REALSENSE_INC}
 	${OpenCV_INCLUDE_DIRS}
   	${ROOT}
)


#文件操作命令，第一个变量为GLOB时，将满足第三个变量条件（“.cpp”）的文件，放入第二个变量中（SOURCES）
file(GLOB SOURCES
    "*.cpp"
    )


#链接库文件
set(LIB_REALSENSE 	"/usr/lib/x86_64-linux-gnu/librealsense2.so")
set( PROJECT_LINK_LIBS
    ${LIB_REALSENSE}
    ${OpenCV_LIBRARIES}
)


#OpenCV_LIBS等同于OpenCV_LIBRARIES
message(STATUS "version: ${OpenCV_VERSION}")
message(STATUS "libraries: ${OpenCV_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")

#生成可执行文件
add_executable(detect_circle ${SOURCES})
target_link_libraries(detect_circle ${PROJECT_LINK_LIBS} ${PCL_LIBRARIES} librealsense2.so)








