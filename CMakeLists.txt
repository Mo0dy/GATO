#
# mkdir build && cd build cmake .. cmake --build . --parallel
#

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "61-real;75-real;86-real;89-real;89-virtual")
endif()

project(GATO LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG
    "-O0 -g -G -fPIC -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -use_fast_math -O3")
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDA REQUIRED)

message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")

include_directories(${CMAKE_SOURCE_DIR}/gato)

# output directory for executables (relative to build directory)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Executable Targets ---
add_executable(bsqp examples/bsqp.cu)
target_compile_definitions(bsqp PRIVATE KNOT_POINTS=16 PLANT_INDY7=1)

if(BUILD_TESTING)
  add_executable(cuda_access_smoke tests/cpp/cuda_access_smoke.cu)
  target_include_directories(cuda_access_smoke PRIVATE ${CMAKE_SOURCE_DIR}/gato
                                                       ${CUDA_INCLUDE_DIRS})
  target_link_libraries(cuda_access_smoke PRIVATE ${CUDA_LIBRARIES})
  add_test(NAME cuda_access_smoke COMMAND cuda_access_smoke)
  set_tests_properties(cuda_access_smoke PROPERTIES LABELS "cuda;smoke")

  add_test(
    NAME python_behavior
    COMMAND ${Python3_EXECUTABLE} -m pytest -m "not cuda and not tracking and not performance" tests/python)
  set_tests_properties(
    python_behavior
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      LABELS "python;smoke")

  add_test(
    NAME python_tracking_smoke
    COMMAND ${Python3_EXECUTABLE} -m pytest -m "tracking and not performance" tests/python)
  set_tests_properties(
    python_tracking_smoke
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      LABELS "python;cuda;solver;tracking"
      DEPENDS cuda_access_smoke
      ENVIRONMENT "GATO_RUN_TRACKING_TESTS=1")
endif()

# --- Python Bindings ---

# horizon length and plant options (configurable via -D at configure time)
set(KNOTS_DEFAULT "8;16;32;64;128")
set(PLANT_DEFAULT "indy7;iiwa14")
set(KNOTS "${KNOTS_DEFAULT}" CACHE STRING "Semicolon-separated horizon lengths (e.g., 32;64;128)")
set(PLANT "${PLANT_DEFAULT}" CACHE STRING "Semicolon-separated plant targets (e.g., indy7;iiwa14)")
# Normalize to semicolon lists (accept commas/spaces)
string(REPLACE "," ";" KNOTS "${KNOTS}")
string(REPLACE " " ";" KNOTS "${KNOTS}")
string(REPLACE "," ";" PLANT "${PLANT}")
string(REPLACE " " ";" PLANT "${PLANT}")
message(STATUS "Configuring Python modules for PLANT='${PLANT}' KNOTS='${KNOTS}'")

function(add_gato_python_module plant knot)
  set(MODULE_NAME "bsqpN${knot}_${plant}")
  if(TARGET ${MODULE_NAME})
    return()
  endif()

  message(STATUS "Configuring Python module: ${MODULE_NAME}")
  pybind11_add_module(${MODULE_NAME} SHARED python/bindings.cu)
  install(
    TARGETS ${MODULE_NAME}
    DESTINATION .
    COMPONENT python)

  target_include_directories(${MODULE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/gato
                                                    ${CUDA_INCLUDE_DIRS})

  # Define plant type at compile time
  if(plant STREQUAL "indy7")
    target_compile_definitions(${MODULE_NAME} PRIVATE KNOT_POINTS=${knot} PLANT_INDY7=1)
  elseif(plant STREQUAL "iiwa14")
    target_compile_definitions(${MODULE_NAME} PRIVATE KNOT_POINTS=${knot} PLANT_IIWA14=1)
  else()
    message(FATAL_ERROR "Unsupported PLANT value for Python module: ${plant}")
  endif()

  set_target_properties(
    ${MODULE_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                              ${CMAKE_SOURCE_DIR}/python/bsqp)
  target_link_libraries(${MODULE_NAME} PRIVATE ${CUDA_LIBRARIES})
endfunction()

foreach(plant ${PLANT})
  foreach(knot ${KNOTS})
    add_gato_python_module(${plant} ${knot})
  endforeach()
endforeach()

if(BUILD_TESTING)
  if(NOT TARGET bsqpN8_indy7)
    message(STATUS "Configuring test-required Python module: bsqpN8_indy7")
    add_gato_python_module(indy7 8)
  endif()

  add_custom_target(
    check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS cuda_access_smoke bsqpN8_indy7
    USES_TERMINAL
    COMMENT "Building test dependencies and running the default test suite")

  add_custom_target(
    check-fast
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R python_behavior
    USES_TERMINAL
    COMMENT "Running fast behavior tests")

  add_custom_target(
    check-python
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R python_behavior
    USES_TERMINAL
    COMMENT "Running fast Python behavior tests")

  add_custom_target(
    check-tracking
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L tracking
    DEPENDS cuda_access_smoke bsqpN8_indy7
    USES_TERMINAL
    COMMENT "Building tracking test dependencies and running tracking tests")

  add_custom_target(
    check-performance
    COMMAND ${CMAKE_COMMAND} -E env
            GATO_RUN_TRACKING_TESTS=1
            GATO_RUN_PERFORMANCE_TESTS=1
            GATO_TEST_ARTIFACT_DIR=${CMAKE_SOURCE_DIR}/test-artifacts/tracking
            ${Python3_EXECUTABLE} -m pytest -m performance tests/python
    DEPENDS cuda_access_smoke bsqpN8_indy7
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    USES_TERMINAL
    COMMENT "Building performance test dependencies and writing tracking artifacts")
endif()
