# CMake package of the MorphIt C library (release archive morphit-capi-<version>-<target>).
#
#   list(APPEND CMAKE_PREFIX_PATH "/path/to/morphit-capi-<version>-<target>")
#   find_package(morphit 0.1 CONFIG REQUIRED)
#   target_link_libraries(app PRIVATE morphit::morphit)          # shared library
#   target_link_libraries(app PRIVATE morphit::morphit_static)   # or the static one
#
# Targets:
#   morphit::morphit         shared library (morphit_capi.dll / libmorphit_capi.so / .dylib)
#   morphit::morphit_static  static library with the system libraries it needs
# Both carry the include directory with morphit.h. On Windows the DLL is in
# bin/; copy it next to your executables, for example with
#   add_custom_command(TARGET app POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      $<TARGET_RUNTIME_DLLS:app> $<TARGET_FILE_DIR:app> COMMAND_EXPAND_LISTS)
# The Windows static library is built against the DLL C runtime (/MD).

cmake_minimum_required(VERSION 3.16)

get_filename_component(_morphit_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(morphit_INCLUDE_DIR "${_morphit_prefix}/include")

# MORPHIT_NATIVE_STATIC_LIBS: what rustc reports the static library needs.
include("${CMAKE_CURRENT_LIST_DIR}/morphit-native-libs.cmake")

if(NOT TARGET morphit::morphit)
  add_library(morphit::morphit SHARED IMPORTED)
  set_target_properties(morphit::morphit PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${morphit_INCLUDE_DIR}")
  if(WIN32)
    set_target_properties(morphit::morphit PROPERTIES
      IMPORTED_LOCATION "${_morphit_prefix}/bin/morphit_capi.dll"
      IMPORTED_IMPLIB "${_morphit_prefix}/lib/morphit_capi.dll.lib")
  elseif(APPLE)
    set_target_properties(morphit::morphit PROPERTIES
      IMPORTED_LOCATION "${_morphit_prefix}/lib/libmorphit_capi.dylib"
      IMPORTED_SONAME "@rpath/libmorphit_capi.dylib")
  else()
    # Rust shared libraries carry no SONAME.
    set_target_properties(morphit::morphit PROPERTIES
      IMPORTED_LOCATION "${_morphit_prefix}/lib/libmorphit_capi.so"
      IMPORTED_NO_SONAME TRUE)
  endif()
endif()

if(NOT TARGET morphit::morphit_static)
  add_library(morphit::morphit_static STATIC IMPORTED)
  if(WIN32)
    set(_morphit_static "${_morphit_prefix}/lib/morphit_capi.lib")
  else()
    set(_morphit_static "${_morphit_prefix}/lib/libmorphit_capi.a")
  endif()
  set_target_properties(morphit::morphit_static PROPERTIES
    IMPORTED_LOCATION "${_morphit_static}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    INTERFACE_INCLUDE_DIRECTORIES "${morphit_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${MORPHIT_NATIVE_STATIC_LIBS}")
  unset(_morphit_static)
endif()

set(morphit_LIBRARIES morphit::morphit)
unset(_morphit_prefix)
