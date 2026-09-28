# Builds the C and C++ examples (../cpp: the header-only morphit.hpp wrapper,
# plus its test program) against the MorphIt C library, either a release
# package (morphit-capi-<version>-<target> archive, found with find_package)
# or this repository's build:
#
#   # release package
#   cmake -S crates/morphit-capi/examples/c -B build/c-examples -DCMAKE_PREFIX_PATH=/path/to/morphit-capi-<v>-<target>
#   # repository build
#   cargo build --release -p morphit-capi
#   cmake -S crates/morphit-capi/examples/c -B build/c-examples
#
#   cmake --build build/c-examples --config Release
#
# -DMORPHIT_STATIC=ON links the static library instead of the shared one. On
# Windows the DLL is copied next to the executables.

cmake_minimum_required(VERSION 3.21)
project(morphit_c_examples C CXX)

option(MORPHIT_STATIC "Link the static MorphIt library" OFF)
find_package(Threads REQUIRED)

find_package(morphit 0.1 CONFIG QUIET)
if(morphit_FOUND)
  message(STATUS "MorphIt ${morphit_VERSION} package: ${morphit_DIR}")
  if(MORPHIT_STATIC)
    set(MORPHIT_TARGET morphit::morphit_static)
    set(MORPHIT_CPP_TARGET morphit::morphit_cpp_static)
  else()
    set(MORPHIT_TARGET morphit::morphit)
    set(MORPHIT_CPP_TARGET morphit::morphit_cpp)
  endif()
else()
  get_filename_component(MORPHIT_REPO "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
  set(MORPHIT_INCLUDE_DIR "${MORPHIT_REPO}/crates/morphit-capi/include" CACHE PATH "Directory with morphit.h")
  set(MORPHIT_LIB_DIR "${MORPHIT_REPO}/target/release" CACHE PATH "Directory with the built MorphIt library")
  if(WIN32)
    find_file(MORPHIT_LIB morphit_capi.dll.lib PATHS "${MORPHIT_LIB_DIR}" NO_DEFAULT_PATH)
  else()
    find_library(MORPHIT_LIB morphit_capi PATHS "${MORPHIT_LIB_DIR}" NO_DEFAULT_PATH)
  endif()
  if(NOT MORPHIT_LIB)
    message(FATAL_ERROR "No MorphIt package (set CMAKE_PREFIX_PATH) and no library in ${MORPHIT_LIB_DIR}; "
                        "run `cargo build --release -p morphit-capi`")
  endif()
  add_library(morphit_repo INTERFACE)
  target_include_directories(morphit_repo INTERFACE "${MORPHIT_INCLUDE_DIR}")
  target_link_libraries(morphit_repo INTERFACE "${MORPHIT_LIB}")
  add_library(morphit_repo_cpp INTERFACE)
  target_link_libraries(morphit_repo_cpp INTERFACE morphit_repo)
  target_compile_features(morphit_repo_cpp INTERFACE cxx_std_17)
  set(MORPHIT_TARGET morphit_repo)
  set(MORPHIT_CPP_TARGET morphit_repo_cpp)
endif()

# Put the DLL next to an executable (Windows; elsewhere the rpath finds it).
function(morphit_runtime target)
  if(NOT WIN32 OR MORPHIT_STATIC)
    return()
  endif()
  if(morphit_FOUND)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
      COMMAND_EXPAND_LISTS)
  elseif(EXISTS "${MORPHIT_LIB_DIR}/morphit_capi.dll")
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MORPHIT_LIB_DIR}/morphit_capi.dll" $<TARGET_FILE_DIR:${target}>)
  endif()
endfunction()

foreach(example pack threads)
  add_executable(${example} ${example}.c)
  target_link_libraries(${example} PRIVATE ${MORPHIT_TARGET} Threads::Threads)
  morphit_runtime(${example})
endforeach()

# C++ (morphit.hpp): pack_cpp, robot_cpp and the wrapper's test program.
foreach(example pack robot test_morphit)
  set(target ${example})
  if(NOT example STREQUAL "test_morphit")
    set(target ${example}_cpp)
  endif()
  add_executable(${target} ../cpp/${example}.cpp)
  target_link_libraries(${target} PRIVATE ${MORPHIT_CPP_TARGET} Threads::Threads)
  morphit_runtime(${target})
endforeach()
