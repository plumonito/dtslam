##################################
# dtslam project
##################################
cmake_minimum_required(VERSION 3.0)

project(dtslam CXX C)

# The version number.
set (dslam_VERSION_MAJOR 0)
set (dslam_VERSION_MINOR 1)

#############
# Macro
macro(GroupSources curdir)
	file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir} ${PROJECT_SOURCE_DIR}/${curdir}/*)

	foreach(child ${children})
		if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
			GroupSources(${curdir}/${child})
		else()
			string(REPLACE "/" "\\" groupname ${curdir})

			# I would like to call the src root folder in a different name, only in visual studio (not mandatory requirement)

			string(REPLACE "src" "Common" groupname ${groupname})

			source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
	   endif()
	endforeach()
endmacro()

##################################
#General compiler flags
#add_definitions(-Wall) #Enable all warnings always
add_definitions(-D_USE_MATH_DEFINES=1)

if(MINGW OR UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -march=corei7")
    set(CMAKE_ECLIPSE_MAKE_ARGUMENTS -j)
    
    if(CMAKE_BUILD_TYPE STREQUAL "")
        set(CMAKE_BUILD_TYPE Release)
        message(STATUS "No build type specified, assuming release.")
    endif()
elseif(MSVC)
    add_definitions("/MP") #Multiprocessor compilation
    add_definitions("/fp:fast") #Fast floating point
    add_definitions("/arch:SSE2") #Multiprocessor compilation
endif()

#macro for installing executable binary and dependencies
macro(INSTALL_TARGET_WITH_DEPS TARGET_NAME BIN_DIR)
    get_target_property(${TARGET_NAME}_LOCATION ${TARGET_NAME} LOCATION_RELEASE)
    get_filename_component(${TARGET_NAME}_LOCATION ${${TARGET_NAME}_LOCATION} NAME)
    set(${TARGET_NAME}_APP ${CMAKE_INSTALL_PREFIX}/${${TARGET_NAME}_LOCATION})
    set(${TARGET_NAME}_DIR ${BIN_DIR})
    install(TARGETS ${TARGET_NAME} DESTINATION .)
    install(CODE "include(BundleUtilities)\nfixup_bundle(\"${${TARGET_NAME}_APP}\" \"\" \"${${TARGET_NAME}_DIR}\")" COMPONENT Runtime)
endmacro(INSTALL_TARGET_WITH_DEPS)

###############################
# Needed package

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
#find_package(Eigen REQUIRED PATHS ${THIRD_PARTY_PREFIX})
#find_package(OpenCV COMPONENTS core calib3d highgui imgproc features2d video NO_MODULE NO_DEFAULT_PATH PATHS ${THIRD_PARTY_PREFIX})
#find_package(GLOG REQUIRED PATHS ${THIRD_PARTY_PREFIX})
#find_package(gflags REQUIRED PATHS ${THIRD_PARTY_PREFIX})
#find_package(SuiteSparse REQUIRED PATHS ${THIRD_PARTY_PREFIX})
#find_package(CERES REQUIRED PATHS ${THIRD_PARTY_PREFIX})
#find_package(freeglut REQUIRED PATHS ${THIRD_PARTY_PREFIX})
#find_package(GLEW REQUIRED PATHS ${THIRD_PARTY_PREFIX})

add_subdirectory(dtslam) 
add_subdirectory(dtslam_ui) 
add_subdirectory(dtslam_desktop)
#add_subdirectory(tests) 
