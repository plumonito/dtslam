find_package(OpenGL REQUIRED)

cmake_policy(SET CMP0022 NEW)

add_library(glew ${CMAKE_CURRENT_SOURCE_DIR}/../${GLEW_ROOT_DIR}/src/glew.c)
target_include_directories(glew PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../${GLEW_ROOT_DIR}/include> $<INSTALL_INTERFACE:include> ${OPENGL_INCLUDE_DIR})
target_link_libraries(glew PUBLIC ${OPENGL_LIBRARIES})
target_compile_definitions(glew PUBLIC -DGLEW_STATIC=1)

install(TARGETS glew EXPORT glew
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../${GLEW_ROOT_DIR}/include/GL
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

install(EXPORT glew DESTINATION lib/glew FILE glew-config.cmake
        EXPORT_LINK_INTERFACE_LIBRARIES)
