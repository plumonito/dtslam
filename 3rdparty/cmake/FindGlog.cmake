# glog depends on gflags and apparently I must do this...
find_package(gflags REQUIRED NO_MODULE QUIET PATHS "${EP_PREFIX}")

find_package(glog REQUIRED NO_MODULE QUIET PATHS "${EP_PREFIX}")
set(GLOG_FOUND true CACHE BOOLEAN "")
set(GLOG_INCLUDE_DIR ${EP_PREFIX}/include CACHE PATH "")
set(GLOG_INCLUDE_DIRS ${GLOG_INCLUDE_DIR} CACHE STRING "")
set(GLOG_LIBRARY glog CACHE STRING "")
set(GLOG_LIBRARIES ${GLOG_LIBRARY} CACHE STRING "")


