cmake_minimum_required(VERSION 3.12.0)
project(lab06 LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_executable(space-time-adaptivity src/space-time-adaptivity.cpp src/Heat.cpp)
deal_ii_setup_target(space-time-adaptivity)

# # Add a test that runs the executable. This test will be discoverable by CTest.
# deal_ii_add_test(space-time-adaptivity)

# # Add a test specifically for model validation, which can be run with `ctest -R model_validation`
# add_test(NAME model_validation COMMAND space-time-adaptivity)
