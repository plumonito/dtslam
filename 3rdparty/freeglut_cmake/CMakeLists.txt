cmake_minimum_required(VERSION 3.0)

cmake_policy(SET CMP0022 NEW)
include(CheckIncludeFiles)

CHECK_INCLUDE_FILES(sys/ioctl.h HAVE_SYS_IOCTL_H)
CHECK_INCLUDE_FILES(fcntl.h HAVE_FCNTL_H)
CHECK_INCLUDE_FILES(errno.h HAVE_ERRNO_H)
CHECK_INCLUDE_FILES(usbhid.h HAVE_USBHID_H)
CHECK_INCLUDE_FILES(sys/param.h HAVE_SYS_PARAM_H)

if(HAVE_SYS_IOCTL_H)
    add_definitions(-DHAVE_SYS_IOCTL_H=1)
endif()
if(HAVE_FCNTL_H)
    add_definitions(-DHAVE_FCNTL_H=1)
endif()
if(HAVE_ERRNO_H)
    add_definitions(-DHAVE_ERRNO_H=1)
endif()
if(HAVE_SYS_PARAM_H)
    add_definitions(-DHAVE_SYS_PARAM_H=1)
endif()


file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../${FREEGLUT_ROOT_DIR}/src/*.c)
add_library(freeglut STATIC ${SOURCES})
if(WIN32)
    target_link_libraries(freeglut PUBLIC winmm)
endif()

set_target_properties(freeglut PROPERTIES OUTPUT_NAME "freeglut_static")

if(WIN32)
    target_compile_definitions(freeglut PUBLIC -DTARGET_HOST_MS_WINDOWS=1)
elseif(APPLE)
    target_compile_definitions(freeglut PUBLIC -DTARGET_HOST_MAX_OSX=1)
else()
    target_compile_definitions(freeglut PUBLIC -DTARGET_HOST_POSIX_X11=1)
endif()

target_include_directories(freeglut PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../${FREEGLUT_ROOT_DIR}/include> $<INSTALL_INTERFACE:include>)
target_compile_definitions(freeglut PUBLIC -DFREEGLUT_STATIC=1
                                           -DFREEGLUT_LIB_PRAGMAS=0)


install(TARGETS freeglut EXPORT freeglut
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../${FREEGLUT_ROOT_DIR}/include/GL
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

install(EXPORT freeglut DESTINATION lib/freeglut FILE freeglut-config.cmake
        EXPORT_LINK_INTERFACE_LIBRARIES)
