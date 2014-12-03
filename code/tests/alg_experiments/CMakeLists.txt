##################################
# test applicaiton
##################################


##################################
# Sources

#Add all files
file(GLOB_RECURSE sources_cpp *.cpp)
file(GLOB_RECURSE sources_h *.h)

##################################
# Target

#add_definitions(-DGLEW_STATIC)
include_directories ("${PROJECT_SOURCE_DIR}/nvslam")
include_directories ("${PROJECT_SOURCE_DIR}/nvslam_ui")

add_executable(alg_experiments ${sources_cpp} ${sources_h})
target_link_libraries (alg_experiments ${gtest_main_LIBRARY} ${gtest_LIBRARY} nvslam)

#INSTALL_TARGET_WITH_DEPS(alg_experiments ${3RD_PARTY_DIR}/bin)

#add_custom_command(TARGET nvslam_desktop_unittest POST_BUILD
#	COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/modules/nvslam/assets $<TARGET_FILE_DIR:nvslam_desktop_unittest>/assets)
#add_custom_command(TARGET nvslam_desktop_unittest POST_BUILD
#	COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/modules/nvslam_app/assets $<TARGET_FILE_DIR:nvslam_desktop_unittest>/assets)

#add_custom_command(TARGET nvslam_desktop_unittest POST_BUILD
#	COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data $<TARGET_FILE_DIR:nvslam_desktop_unittest>/data)
