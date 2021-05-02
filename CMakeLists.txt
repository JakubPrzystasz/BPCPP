cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.0)

set(CMAKE_SUPPRESS_REGENERATION true)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)

project("Backprop CPP" VERSION 1.0)

include(${CMAKE_SOURCE_DIR}/CMake/RedirectOutputTo.cmake)
include(${CMAKE_SOURCE_DIR}/CMake/CompileFlags.cmake)
include(${CMAKE_SOURCE_DIR}/CMake/CreateSymlink.cmake)
include(${CMAKE_SOURCE_DIR}/CMake/GroupSources.cmake)
include(${CMAKE_SOURCE_DIR}/CMake/StaticLibsCompile.cmake)
include(${CMAKE_SOURCE_DIR}/CMake/BuildExamples.cmake)

file(MAKE_DIRECTORY "bin")
RedirectOutputTo("bin")

# Library target
add_library(matplotlib_cpp INTERFACE)

target_include_directories(matplotlib_cpp
  INTERFACE
  ${PROJECT_SOURCE_DIR}/headers/matplotlib
)

target_compile_features(matplotlib_cpp INTERFACE
  cxx_std_11
)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

target_link_libraries(matplotlib_cpp INTERFACE
  Python3::Python
  Python3::Module
)

find_package(Python3 COMPONENTS NumPy)

target_link_libraries(matplotlib_cpp INTERFACE
    Python3::NumPy
)



set(ENGINE_DIR ${PROJECT_SOURCE_DIR}/src/lib)

file(GLOB_RECURSE SRC_FILES 
  ${ENGINE_DIR}/*.cpp
  ${ENGINE_DIR}/*.cxx
  ${ENGINE_DIR}/*.c
)

foreach(FILE ${SRC_FILES})
  message("[Lib]Found source file: ${FILE}")
  set(ENGINE_SRC ${ENGINE_SRC}  ${FILE})
endforeach(FILE)

add_library(Lib STATIC ${ENGINE_SRC})
target_link_libraries(Lib PRIVATE matplotlib_cpp)

set(EngineHeaders 
  ${PROJECT_SOURCE_DIR}/headers/lib
)

target_include_directories(Lib PRIVATE ${EngineHeaders})

set(ENGINE_DIR ${PROJECT_SOURCE_DIR}/headers/lib)

file(GLOB_RECURSE SRC_FILES 
  ${ENGINE_DIR}/*.h
  ${ENGINE_DIR}/*.hpp
  ${ENGINE_DIR}/*.hxx
)

foreach(FILE ${SRC_FILES})
  message("[Lib]Found header file: ${FILE}")
  set(ENGINE_SRC ${ENGINE_SRC}  ${FILE})
endforeach(FILE)

target_sources(Lib PRIVATE ${ENGINE_SRC})
GroupSources("src/lib" "src/lib" "Source Files")


### Executable
set(LAUNCHER_DIR ${PROJECT_SOURCE_DIR}/src/exe)

file(GLOB_RECURSE SRC_FILES 
  ${LAUNCHER_DIR}/*.cpp
  ${LAUNCHER_DIR}/*.cxx
  ${LAUNCHER_DIR}/*.c
)

foreach(FILE ${SRC_FILES})
  message("[exe]Found source file: ${FILE}")
  set(LAUNCHER_SRC ${LAUNCHER_SRC}  ${FILE})
endforeach(FILE)

add_executable(Launcher ${SRC_FILES})
set_property(TARGET Launcher PROPERTY CXX_STANDARD 20)

set(LauncherHeaders 
  ${PROJECT_SOURCE_DIR}/headers
)

target_include_directories(Launcher PRIVATE ${LauncherHeaders})

set(LAUNCHER_DIR ${PROJECT_SOURCE_DIR}/headers)

file(GLOB_RECURSE SRC_FILES 
  ${LAUNCHER_DIR}/exe/*.h
  ${LAUNCHER_DIR}/exe/*.hpp
  ${LAUNCHER_DIR}/exe/*.hxx
)

foreach(FILE ${SRC_FILES})
  message("[exe]Found header file: ${FILE}")
  set(LAUNCHER_SRC ${LAUNCHER_SRC}  ${FILE})
endforeach(FILE)

target_sources(Launcher PRIVATE ${LAUNCHER_SRC})
GroupSources("src/exe" "src/exe" "Source Files")

target_link_libraries(Launcher Lib)
