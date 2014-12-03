cmake_minimum_required(VERSION 3.0)

INCLUDE (CheckIncludeFiles)
INCLUDE (CheckFunctionExists)
INCLUDE (CheckSymbolExists)
INCLUDE (CheckSymbolExists)
INCLUDE (CheckCXXSourceCompiles)
INCLUDE (CheckCSourceCompiles)
INCLUDE (CheckTypeSize)
project(glog)

cmake_policy(VERSION 3.0)


if(NOT HAS_MZ_GLOBAL)
##################################################
#
#	BUILD/GLOBAL.CMAKE
#
# 	This file is for providing a defined environment
#	of compiler definitions/macros and cmake functions
#	or variables throughout several projects. It can
#	be included twice or more without any issues and
#   will automatically included the utility files 
#   compiler.cmake and macros.cmake
#
#	(c) 2009-2012 Marius Zwicker
#
##################################################

### CONFIGURATION SECTION

    cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
# CMAKE_CURRENT_LIST_DIR is available after CMake 2.8.3 only
# but we support 2.8.0 as well
    if( NOT CMAKE_CURRENT_LIST_DIR )
        string(REPLACE "/global.cmake" "" CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}")
    endif()

# path to the mz tools files
    set(MZ_TOOLS_PATH "${CMAKE_CURRENT_LIST_DIR}")

### END OF CONFIGURATION SECTION

# BOF: global.cmake
    if(NOT HAS_MZ_GLOBAL)
            set(HAS_MZ_GLOBAL true)

      # detect compiler
      include("${MZ_TOOLS_PATH}/compiler.cmake")

      # user info
      message("-- configuring for build type: ${CMAKE_BUILD_TYPE}")
      
      # macros
      include("${MZ_TOOLS_PATH}/macros.cmake")

# EOF: global.cmake
    endif() 
endif()

CHECK_INCLUDE_FILES (dlfcn.h HAVE_DLFCN_H)
CHECK_INCLUDE_FILES (execinfo.h HAVE_EXECINFO_H)
CHECK_INCLUDE_FILES (glob.h HAVE_GLOB_H)
CHECK_INCLUDE_FILES (inttypes.h HAVE_INTTYPES_H)
CHECK_INCLUDE_FILES (pthread.h HAVE_LIBPTHREAD)
CHECK_INCLUDE_FILES (libunwind.h HAVE_LIBUNWIND_H)
CHECK_INCLUDE_FILES (libunwind.h HAVE_LIB_UNWIND)
set(HAVE_LIB_GFLAGS TRUE) # we can be sure as this is a build dependency
CHECK_INCLUDE_FILES (memory.h HAVE_MEMORY_H)
CHECK_INCLUDE_FILES (pthread.h HAVE_PTHREAD_H)
CHECK_INCLUDE_FILES (pwd.h HAVE_PWD_H)
CHECK_INCLUDE_FILES (stdint.h HAVE_STDINT_H)
CHECK_INCLUDE_FILES (stdlib.h HAVE_STDLIB_H)
CHECK_INCLUDE_FILES (strings.h HAVE_STRINGS_H)
CHECK_INCLUDE_FILES (syscall.h HAVE_SYSCALL_H)
CHECK_INCLUDE_FILES (syslog.h HAVE_SYSLOG_H)
CHECK_INCLUDE_FILES (sys/stat.h HAVE_SYS_STAT_H)
CHECK_INCLUDE_FILES (sys/call.h HAVE_SYS_CALL_H)
CHECK_INCLUDE_FILES (sys/time.h HAVE_SYS_TIME_H)
CHECK_INCLUDE_FILES (sys/types.h HAVE_SYS_TYPES_H)
CHECK_INCLUDE_FILES (sys/syscall.h HAVE_SYS_SYSCALL_H)
CHECK_INCLUDE_FILES (sys/ucontext.h HAVE_SYS_UCONTEXT_H)
CHECK_INCLUDE_FILES (sys/utsname.h HAVE_SYS_UTSNAME_H)
CHECK_INCLUDE_FILES (ucontext.h HAVE_UCONTEXT_H)
CHECK_INCLUDE_FILES (unistd.h HAVE_UNISTD_H)
CHECK_INCLUDE_FILES (unwind.h HAVE_UNWIND_H)
CHECK_INCLUDE_FILES (inttypes.h HAVE_INTTYPES_H)

CHECK_FUNCTION_EXISTS(fcntl HAVE_FCNTL)
CHECK_FUNCTION_EXISTS(pthread_rwlock_rdlock HAVE_RWLOCK)
CHECK_FUNCTION_EXISTS(sigaltstack HAVE_SIGALTSTACK)
CHECK_FUNCTION_EXISTS(__builtin_expect HAVE___BUILTIN_EXPECT)
CHECK_FUNCTION_EXISTS(__sync_val_compare_and_swap HAVE___SYNC_VAL_COMPARE_AND_SWAP)

CHECK_TYPE_SIZE(uint16_t HAVE_UINT16_T)
CHECK_TYPE_SIZE(u_int16_t HAVE_U_INT16_T)
CHECK_TYPE_SIZE(__int16 HAVE___INT16)

CHECK_CXX_SOURCE_COMPILES("int main() { if(__builtin_expect(1,1)); return 0; }" HAVE___BUILTIN_EXPECT)
CHECK_CXX_SOURCE_COMPILES("namespace foo {int a;} using foo::a; int main() { return 0; }" HAVE_CXX_USING_OPERATOR)
CHECK_CXX_SOURCE_COMPILES("namespace foo {} int main() { return 0; }" HAVE_NAMESPACES)
CHECK_CXX_SOURCE_COMPILES("int main() { int val; __sync_val_compare_and_swap(&val, 1, 1); return 0; }" HAVE___SYNC_VAL_COMPARE_AND_SWAP)

if(NOT WIN32)
    if(NOT PC_FROM_UCONTEXT)
        foreach(UCONTEXT_HEADER ucontext.h sys/ucontext.h)
            foreach(pc_ucontext 
               "uc_mcontext.gregs[REG_PC]"             # Solaris x86 (32 + 64bits) 
               "uc_mcontext.gregs[REG_EIP]"            # Linux (i386)
               "uc_mcontext.gregs[REG_RIP]"            # Linux (x86_64)
               "uc_mcontext.sc_ip"                     # Linux (ia64)
               "uc_mcontext.uc_regs->gregs[PT_NIP]"    # Linux (ppc)
               "uc_mcontext.gregs[R15]"                # Linux (arm old [untested])
               "uc_mcontext.arm_pc"                    # Linux (arm new [untested])
               "uc_mcontext.mc_eip"                    # FreeBSD (i386)
               "uc_mcontext.mc_rip"                    # FreeBSD (x86_64 [untested])
               "uc_mcontext.__gregs[_REG_EIP]"         # NetBSD (i386)
               "uc_mcontext.__gregs[_REG_RIP]"         # NetBSD (x86_64)
               "uc_mcontext->ss.eip"                   # OS X (i386, <=10.4)
               "uc_mcontext->__ss.__eip"               # OS X (i386, >=10.5)
               "uc_mcontext->ss.rip"                   # OS X (x86_64)
               "uc_mcontext->__ss.__rip"               # OS X (>=10.5 [untested])
               "uc_mcontext->ss.srr0"                  # OS X (ppc, ppc64 [untested])
               "uc_mcontext->__ss.__srr0")             # OS X (>=10.5 [untested])

               unset(PC_FROM_UCONTEXT CACHE)

                   CHECK_C_SOURCE_COMPILES(
                "
                    #define _GNU_SOURCE 1
                    #include <${UCONTEXT_HEADER}>
                    int main()
                    {
                        ucontext_t u;
                        return u.${pc_ucontext} == 0;
                    }
                    " PC_FROM_UCONTEXT)

                if(PC_FROM_UCONTEXT)
                    set(PC_FROM_UCONTEXT "${pc_ucontext}" CACHE STRING "" FORCE)
                    break()
                endif()
            endforeach()
            if(PC_FROM_UCONTEXT)
                break()
            endif()
        endforeach()

        if(NOT PC_FROM_UCONTEXT)
            foreach(pc_ucontext sc_eip sc_rip)
                unset(PC_FROM_UCONTEXT CACHE)
                CHECK_C_SOURCE_COMPILES(
                    " #define _GNU_SOURCE 1
                      #include <signal.h>
                      int main()
                      {
                        ucontext_t u;
                        return u.${pc_ucontext} == 0;
                      }
                    " PC_FROM_UCONTEXT)
                if(PC_FROM_UCONTEXT)
                    set(PC_FROM_UCONTEXT ${pc_ucontext})
                    break()
                endif()
            endforeach()
        endif()
    endif()
endif()

if(HAVE_UNISTD_H)
    set(ac_cv_have_unistd_h 1)
else()
    set(ac_cv_have_unistd_h 0)
endif()

if(HAVE_STDINT_H)
    set(ac_cv_have_stdint_h 1)
else()
    set(ac_cv_have_stdint_h 0)
endif()

if(HAVE_SYS_TYPES_H)
    set(ac_cv_have_systypes_h 1)
else()
    set(ac_cv_have_systypes_h 0)
endif()

if(HAVE_INTTYPES_H)
    set(ac_cv_have_inttypes_h 1)
else()
    set(ac_cv_have_inttypes_h 0)
endif()

if(HAVE_UINT16_T)
    set(ac_cv_have_uint16_t 1)
else()
    set(ac_cv_have_uint16_t 0)
endif()

if(HAVE_U_INT16_T)
    set(ac_cv_have_u_int16_t 1)
else()
    set(ac_cv_have_u_int16_t 0)
endif()

if(HAVE___UINT16)
    set(ac_cv_have___uint16 1)
else()
    set(ac_cv_have___uint16 0)
endif()

set(ac_cv_have_libgflags 1)

if(HAVE___BUILTIN_EXPECT)
    set(ac_cv_have___builtin_expect 1)
else()
    set(ac_cv_have___builtin_expect 0)
endif()

if(HAVE_CXX_USING_OPERATOR)
    set(ac_cv_cxx_using_operator 1)
else()
    set(ac_cv_cxx_using_operator 0)
endif()

if(NOT HAVE_NAMESPACES)
    set(HAVE_NAMESPACES 0)
endif()

if(NOT HAVE___SYNC_VAL_COMPARE_AND_SWAP)
    set(HAVE___SYNC_VAL_COMPARE_AND_SWAP 0)
endif()

set(ac_cv___attribute___noreturn)
set(ac_cv___attribute___printf_4_5)

set(ac_google_start_namespace "namespace google {")
set(ac_google_end_namespace "}")
set(ac_google_namespace "google")
set(SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})

set(_START_GOOGLE_NAMESPACE_ "${ac_google_start_namespace}")
set(_END_GOOGLE_NAMESPACE_ "${ac_google_end_namespace}")
set(GOOGLE_NAMESPACE "${ac_google_namespace}")
set(STDC_HEADERS 1)
set(STL_NAMESPACE std)

set(PACKAGE "glog")
set(PACKAGE_NAME "glog")


set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../${GLOG_ROOT_DIR})

CONFIGURE_FILE(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/../${GLOG_ROOT_DIR}/src/config.h)
CONFIGURE_FILE(${SRCDIR}/src/glog/logging.h.in ${CMAKE_CURRENT_BINARY_DIR}/../${GLOG_ROOT_DIR}/src/glog/logging.h)
CONFIGURE_FILE(${SRCDIR}/src/glog/raw_logging.h.in ${CMAKE_CURRENT_BINARY_DIR}/../${GLOG_ROOT_DIR}/src/glog/raw_logging.h)
CONFIGURE_FILE(${SRCDIR}/src/glog/stl_logging.h.in ${CMAKE_CURRENT_BINARY_DIR}/../${GLOG_ROOT_DIR}/src/glog/stl_logging.h)
CONFIGURE_FILE(${SRCDIR}/src/glog/vlog_is_on.h.in ${CMAKE_CURRENT_BINARY_DIR}/../${GLOG_ROOT_DIR}/src/glog/vlog_is_on.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../${GLOG_ROOT_DIR}/src)


mz_use_default_compiler_settings()
mz_add_flag(GCC -fPIC)

if(WIN32)
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../${GLOG_ROOT_DIR}/src/windows)

	mz_add_definition(GOOGLE_GLOG_DLL_DECL)
	set(SRCS
		${SRCDIR}/src/logging.cc
		${SRCDIR}/src/windows/port.cc
		${SRCDIR}/src/raw_logging.cc
		${SRCDIR}/src/utilities.cc
		${SRCDIR}/src/vlog_is_on.cc
	)
	
	set(HDRS
		${SRCDIR}/src/base/commandlineflags.h
		${SRCDIR}/src/windows/config.h
		${SRCDIR}/src/base/googleinit.h
		${SRCDIR}/src/windows/glog/log_severity.h
		${SRCDIR}/src/windows/glog/logging.h
		${SRCDIR}/src/base/mutex.h
		${SRCDIR}/src/windows/port.h
		${SRCDIR}/src/windows/glog/raw_logging.h
		${SRCDIR}/src/windows/glog/stl_logging.h
		${SRCDIR}/src/utilities.h
		${SRCDIR}/src/windows/glog/vlog_is_on.h
	)

	include_directories(
            ${SRCDIR}/windows/
		${GFLAGS_INCLUDE_DIRS}
	)

	mz_add_definition(GLOG_NO_ABBREVIATED_SEVERITIES)
	mz_add_definition(GOOGLE_GLOG_DLL_DECL=)
else()
	set(SRCS
            ${SRCDIR}/src/logging.cc
            ${SRCDIR}/src/raw_logging.cc
            ${SRCDIR}/src/vlog_is_on.cc
            ${SRCDIR}/src/utilities.cc
            ${SRCDIR}/src/utilities.h
            ${SRCDIR}/src/demangle.cc ${SRCDIR}/src/demangle.h
            ${SRCDIR}/src/stacktrace.h
            ${SRCDIR}/src/stacktrace_generic-inl.h
            ${SRCDIR}/src/stacktrace_libunwind-inl.h
            ${SRCDIR}/src/stacktrace_powerpc-inl.h
            ${SRCDIR}/src/stacktrace_x86-inl.h
            ${SRCDIR}/src/stacktrace_x86_64-inl.h
            ${SRCDIR}/src/symbolize.cc ${SRCDIR}/src/symbolize.h
            ${SRCDIR}/src/signalhandler.cc
            ${SRCDIR}/src/base/mutex.h ${SRCDIR}/src/base/googleinit.h
            ${SRCDIR}/src/base/commandlineflags.h ${SRCDIR}/src/googletest.h
	)
	
	set(HDRS
		${SRCDIR}/src/glog/log_severity.h
		${SRCDIR}/src/glog/logging.h 
		${SRCDIR}/src/glog/raw_logging.h 
		${SRCDIR}/src/glog/vlog_is_on.h 
		${SRCDIR}/src/glog/stl_logging.h
	)

	include_directories(
            ${SRCDIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/../${GLOG_ROOT_DIR}/src
            ${GFLAGS_INCLUDE_DIRS}
	)

	mz_add_definition(NO_THREADS)
	set(GLOG_3LIBS pthread)
        if( HAVE_LIB_UNWIND )
            set(GLOG_3LIBS ${GLOG_3LIBS} unwind)
        endif()
endif()

find_package(gflags REQUIRED)

cmake_policy(SET CMP0022 NEW)

mz_add_flag(GCC -Wall -Wwrite-strings -Woverloaded-virtual -Wno-sign-compare)
add_library(glog STATIC ${SRCS} ${HDRS})
target_link_libraries(glog ${gflags_LIBRARIES} ${GLOG_3LIBS})
message("GFLAGS LIBS ${gflags_LIBRARIES} ------------------------------------")
if(WIN32)
    # so that we can link against the static library we're creating
    target_compile_definitions(glog INTERFACE -DGOOGLE_GLOG_DLL_DECL=)
endif()
include_directories(${gflags_INCLUDE_DIR})
set_property(TARGET glog PROPERTY VERSION 0.3.3)
set_property(TARGET glog PROPERTY SOVERSION 0.3.3)
mz_target_props(glog)

if(WIN32)
    install(TARGETS glog
                EXPORT glog
                LIBRARY DESTINATION lib
                RUNTIME DESTINATION bin
                ARCHIVE DESTINATION lib)
    install(DIRECTORY ${SRCDIR}/src/windows/glog
                DESTINATION include
                PATTERN "*.svn" EXCLUDE)
    install(DIRECTORY ${SRCDIR}/doc
                DESTINATION doc
                PATTERN "*.svn" EXCLUDE)
else()
    install(TARGETS glog
                EXPORT glog
                LIBRARY DESTINATION lib
                RUNTIME DESTINATION bin
                ARCHIVE DESTINATION lib)
    install(FILES ${SRCDIR}/src/glog/log_severity.h
                DESTINATION include/glog)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../${GLOG_ROOT_DIR}/src/
                DESTINATION include
                PATTERN "*.svn" EXCLUDE)
    if(APPLE)
        install(DIRECTORY ${SRCDIR}/doc
                DESTINATION Library/Documentation/glog
                PATTERN "*.svn" EXCLUDE)
    else()
        install(DIRECTORY ${SRCDIR}/doc
                DESTINATION share/doc/glog-doc
                PATTERN "*.svn" EXCLUDE)
    endif()
endif()

install(EXPORT glog FILE glog-config.cmake DESTINATION lib/glog
        EXPORT_LINK_INTERFACE_LIBRARIES)
        
