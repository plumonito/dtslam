macro(ConfigExternalProject NAME)
    set(EP_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/usr" CACHE PATH "")

    file(MAKE_DIRECTORY "${EP_PREFIX}")

    set(USER_MODULE_PATH)

    set(params)
    set(is_value false)
    set(param)
    foreach(arg ${ARGN})
        #message("ARG: ${arg}")
        if("${arg}" STREQUAL "MODULE_PATH")
            set(is_value true)
            set(param "${arg}")
        elseif(is_value)
            if("${param}" STREQUAL "MODULE_PATH")
                set(USER_MODULE_PATH ${arg})
            endif()
            set(is_value false)
        else()
            string(REPLACE " " "\\ " arg "${arg}")
            set(params "${params} ${arg}")
        endif()
    endforeach()

    if(MSVC)
        set(BUILD_TYPES Debug Release)
    else()
        set(BUILD_TYPES ${CMAKE_BUILD_TYPE})
    endif()

    if(BUILD_TYPES STREQUAL "")
        set(BUILD_TYPES Release)
    endif()

    foreach(BUILD_TYPE ${BUILD_TYPES})
        if(NV_${NAME}_${BUILD_TYPE}_LOCALLY_INSTALLED)
            message(STATUS "3rd-party ${NAME}-${BUILD_TYPE} library already installed")
        else()
            set(bindir "${CMAKE_CURRENT_BINARY_DIR}/${NAME}/${BUILD_TYPE}")
            file(MAKE_DIRECTORY "${bindir}")
            set(tmpfile "${bindir}/CMakeLists.txt")

            if(MSVC)
                # we pass the default C(XX) flags and the ones (possibly) defined by the user
                set(parallel_cl_cxx_build -DCMAKE_CXX_FLAGS:STRING=\${CMAKE_CXX_FLAGS}\\\ \\\${CMAKE_CXX_FLAGS}\\\ /MP)
                set(parallel_cl_c_build -DCMAKE_C_FLAGS:STRING=\${CMAKE_C_FLAGS}\\\ \\\${CMAKE_C_FLAGS}\\\ /MP)
            else()
                # we pass the default C(XX) flags and the ones (possibly) defined by the user
                set(parallel_cl_cxx_build -DCMAKE_CXX_FLAGS:STRING=\${CMAKE_CXX_FLAGS}\\\ \\\${CMAKE_CXX_FLAGS})
                set(parallel_cl_c_build -DCMAKE_C_FLAGS:STRING=\${CMAKE_C_FLAGS}\\\ \\\${CMAKE_C_FLAGS})
            endif()

            set(USER_MODULE_PATH "${USER_MODULE_PATH}\\\;${CMAKE_CURRENT_SOURCE_DIR}/cmake")

            file(WRITE "${tmpfile}" 
                "cmake_minimum_required(VERSION 3.0)\n"
                "include(ExternalProject)\n"
                "ExternalProject_Add(${NAME} ${params}
                                     INSTALL_DIR \"${EP_PREFIX}\"
                                     CMAKE_ARGS -DCMAKE_PREFIX_PATH:PATH=${EP_PREFIX} -DCMAKE_INSTALL_PREFIX:PATH=${EP_PREFIX} -Wno-dev -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_DEBUG_POSTFIX:STRING=d
                                     CMAKE_CACHE_ARGS ${parallel_cl_cxx_build} ${parallel_cl_c_build} -DEP_PREFIX:PATH=${EP_PREFIX} -DCMAKE_MODULE_PATH:STRING=${USER_MODULE_PATH} 
                                     PREFIX \"${bindir}\")\n")

            message(STATUS "Configuring ${NAME} - ${BUILD_TYPE}")

            execute_process(COMMAND ${CMAKE_COMMAND} "${bindir}"
                            WORKING_DIRECTORY "${bindir}"
                            RESULT_VARIABLE result)
                            #                    OUTPUT_FILE ${bindir}/config_output.txt
                            #ERROR_FILE ${bindir}/config_error.txt
                            #                    OUTPUT_VARIABLE errmsg
                            #ERROR_VARIABLE errmsg)
            if(NOT "${result}" STREQUAL "0")
                message(FATAL_ERROR "Error configuring ${NAME}: ${result} ${errmsg}")
            endif()

            message(STATUS "Building ${NAME} - ${BUILD_TYPE}")

            if(MSVC)
                set(PARALLEL_BUILD /maxcpucount)
            else()
                include(ProcessorCount)
                ProcessorCount(N)
                if(NOT N EQUAL 0)
                    set(PARALLEL_BUILD -j${N})
                endif()
            endif()

            execute_process(COMMAND ${CMAKE_COMMAND} --build . --config ${BUILD_TYPE} -- ${PARALLEL_BUILD}
                            WORKING_DIRECTORY "${bindir}"
                            RESULT_VARIABLE result
                            OUTPUT_FILE ${bindir}/build_output.txt
                            ERROR_FILE ${bindir}/build_error.txt)
                            #OUTPUT_VARIABLE errmsg
                            #ERROR_VARIABLE errmsg)

            if(NOT "${result}" STREQUAL "0")
                message(FATAL_ERROR "Error building ${NAME}: ${result} ${errmsg}")
            endif()

            set(NV_${NAME}_${BUILD_TYPE}_LOCALLY_INSTALLED true CACHE BOOLEAN "")
        endif()
    endforeach()

    find_package(${NAME} QUIET NO_DEFAULT_PATH PATHS "${EP_PREFIX}")
    if(${NAME}_FOUND)
        message(STATUS "Using built-in ${NAME} library")
    endif()

    include_directories(${EP_PREFIX}/include)
endmacro()
