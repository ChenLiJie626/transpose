cmake_minimum_required(VERSION 3.5.1)

project(acl_execute_complex_transpose)

add_compile_options(-std=c++11 -O2)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../output")

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")
set(ASCEND_DEVLIB_PATH "${INC_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib")
set(ASCEND_DRIVER_LIB_PATH "/usr/local/Ascend/driver/lib64/driver")

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME_LOWER)
    set(LIB_PATH "${INC_PATH}/${CMAKE_SYSTEM_PROCESSOR}-${SYSTEM_NAME_LOWER}/lib64")
endif()

include_directories(
    ../inc
    ${INC_PATH}/include
    ${CUST_PKG_PATH}/include
)

link_directories(
    ${LIB_PATH}
    ${ASCEND_DEVLIB_PATH}
    ${ASCEND_DRIVER_LIB_PATH}
    ${CUST_PKG_PATH}/lib
)

add_executable(execute_complex_transpose main.cpp)

target_link_libraries(execute_complex_transpose
    ascendcl
    cust_opapi
    acl_op_compiler
    nnopbase
    ascend_hal
    stdc++
)
