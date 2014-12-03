###################################
# Dependencies

#gtest
find_path(gtest_INCLUDE_DIR gtest/gtest.h HINTS "${3RD_PARTY_DIR}/gtest-1.7.0/include")
find_library(gtest_LIBRARY gtest HINTS "${3RD_PARTY_DIR}/gtest-1.7.0/lib")
find_library(gtest_main_LIBRARY gtest_main HINTS "${3RD_PARTY_DIR}/gtest-1.7.0/lib")

#includes
include_directories(${NVSLAM_DEPENDENCY_DIRS})
include_directories(${gtest_INCLUDE_DIR})

#Other test programs
#add_subdirectory(alg_experiments) 

#Add files
list(APPEND sources_cpp "CameraModelTests.cpp")
list(APPEND sources_cpp "PoseEstimationTests.cpp")

##################################
# Target

#include_directories ("${PROJECT_SOURCE_DIR}/nvslam")
#include_directories ("${PROJECT_SOURCE_DIR}/nvslam_ui")

add_executable(nvslam_gtests ${sources_cpp})
target_link_libraries (nvslam_gtests ${gtest_main_LIBRARY} ${gtest_LIBRARY} nvslam)

