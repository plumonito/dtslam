cmake_minimum_required(VERSION 3.0)

project(dtslam CXX C)

add_subdirectory(3rdparty)

if(MSVC)
    set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    set(CMAKE_MAP_IMPORTED_CONFIG Release)
endif()

find_package(freeglut REQUIRED QUIET PATHS "${EP_PREFIX}")
find_package(glew REQUIRED QUIET PATHS "${EP_PREFIX}")
find_package(OpenCV REQUIRED QUIET PATHS "${EP_PREFIX}")
find_package(gflags REQUIRED PATHS "${EP_PREFIX}")
find_package(glog REQUIRED PATHS "${EP_PREFIX}")

if(WIN32)
    set(SuiteSparse_USE_LAPACK_BLAS ON)
endif()
find_package(SuiteSparse REQUIRED PATHS "${EP_PREFIX}")

find_package(Ceres REQUIRED QUIET PATHS "${EP_PREFIX}")

add_definitions(-DGOOGLE_GLOG_DLL_DECL= -DGLEW_STATIC=1 -DFREEGLUT_STATIC=1
    -DFREEGLUT_LIB_PRAGMAS=0)

add_subdirectory(code)
