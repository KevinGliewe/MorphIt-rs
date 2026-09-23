# Builds the C examples against the MorphIt C library, either a release
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
# -DMORPHIT_STATIC=ON links the static library instead of the shared one. With
# the package the DLL is copied next to the executables on Windows; with the
# repository build, copy target/release/morphit_capi.dll or put it on PATH.

cmake_minimum_required(VERSION 3.21)
project(morphit_c_examples C)

option(MORPHIT_STATIC "Link the static MorphIt library" OFF)
find_package(Threads REQUIRED)

find_package(morphit 0.1 CONFIG QUIET)
if(morphit_FOUND)
  message(STATUS "MorphIt ${morphit_VERSION} package: ${morphit_DIR}")
  if(MORPHIT_STATIC)
    set(MORPHIT_TARGET morphit::morphit_static)
  else()
    set(MORPHIT_TARGET morphit::morphit)
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
  set(MORPHIT_TARGET morphit_repo)
endif()

foreach(example pack threads)
  add_executable(${example} ${example}.c)
  target_link_libraries(${example} PRIVATE ${MORPHIT_TARGET} Threads::Threads)
  if(WIN32 AND morphit_FOUND AND NOT MORPHIT_STATIC)
    add_custom_command(TARGET ${example} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${example}> $<TARGET_FILE_DIR:${example}>
      COMMAND_EXPAND_LISTS)
  endif()
endforeach()
