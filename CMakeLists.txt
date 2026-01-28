cmake_minimum_required(VERSION 3.15...3.31)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# macOS frameworks
find_library(CORESERVICES_FRAMEWORK CoreServices REQUIRED)
find_library(COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
find_library(AUDIOUNIT_FRAMEWORK AudioUnit REQUIRED)
find_library(AUDIOTOOLBOX_FRAMEWORK AudioToolbox REQUIRED)
find_library(COREAUDIO_FRAMEWORK CoreAudio REQUIRED)
find_library(COREMIDI_FRAMEWORK CoreMIDI REQUIRED)

# Ableton Link include directories
set(LINK_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/link/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/link/modules/asio-standalone/asio/include
)

# Cython compiler directives
set(CYTHON_DIRECTIVES -3 --directive embedsignature=True)

# ----------------------------------------------------------------------------
# capi module (C extension for CoreAudio/CoreMIDI)
# ----------------------------------------------------------------------------
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/capi.c
    COMMAND Python::Interpreter -m cython
        ${CYTHON_DIRECTIVES}
        "${CMAKE_CURRENT_SOURCE_DIR}/src/coremusic/capi.pyx"
        --output-file ${CMAKE_CURRENT_BINARY_DIR}/capi.c
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/coremusic/capi.pyx
        ${CMAKE_CURRENT_SOURCE_DIR}/src/coremusic/coreaudio.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/src/coremusic/coremidi.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/src/coremusic/coreaudiotypes.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/src/coremusic/corefoundation.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/src/coremusic/audiotoolbox.pxd
    COMMENT "Cythonizing capi.pyx"
)

python_add_library(capi MODULE ${CMAKE_CURRENT_BINARY_DIR}/capi.c WITH_SOABI)
target_compile_definitions(capi PRIVATE LINK_PLATFORM_MACOSX=1)
target_compile_options(capi PRIVATE -Wall -Wextra)
target_link_libraries(capi PRIVATE
    ${CORESERVICES_FRAMEWORK}
    ${COREFOUNDATION_FRAMEWORK}
    ${AUDIOUNIT_FRAMEWORK}
    ${AUDIOTOOLBOX_FRAMEWORK}
    ${COREAUDIO_FRAMEWORK}
    ${COREMIDI_FRAMEWORK}
)

# ----------------------------------------------------------------------------
# link module (C++ extension for Ableton Link)
# ----------------------------------------------------------------------------
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/link.cpp
    COMMAND Python::Interpreter -m cython
        ${CYTHON_DIRECTIVES}
        "${CMAKE_CURRENT_SOURCE_DIR}/src/coremusic/link.pyx"
        --output-file ${CMAKE_CURRENT_BINARY_DIR}/link.cpp
        --cplus
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/coremusic/link.pyx
        ${CMAKE_CURRENT_SOURCE_DIR}/src/coremusic/link.pxd
    COMMENT "Cythonizing link.pyx"
)

python_add_library(link MODULE ${CMAKE_CURRENT_BINARY_DIR}/link.cpp WITH_SOABI)
target_include_directories(link PRIVATE ${LINK_INCLUDE_DIRS})
target_compile_definitions(link PRIVATE LINK_PLATFORM_MACOSX=1)
target_compile_options(link PRIVATE -Wall -Wextra -std=c++11)
target_link_libraries(link PRIVATE
    ${CORESERVICES_FRAMEWORK}
    ${COREFOUNDATION_FRAMEWORK}
    ${AUDIOUNIT_FRAMEWORK}
    ${AUDIOTOOLBOX_FRAMEWORK}
    ${COREAUDIO_FRAMEWORK}
)

# ----------------------------------------------------------------------------
# Install targets
# ----------------------------------------------------------------------------
install(TARGETS capi link DESTINATION coremusic)
