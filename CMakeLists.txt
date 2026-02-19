cmake_minimum_required(VERSION 3.12.0)
project(space-time-adaptivity LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_executable(space-time-adaptivity src/space-time-adaptivity.cpp src/Heat.cpp)
deal_ii_setup_target(space-time-adaptivity)

deal_ii_setup_target(space-time-adaptivity)

enable_testing()

add_test(NAME space-time-adaptivity
         COMMAND space-time-adaptivity)

add_test(NAME model_validation
         COMMAND space-time-adaptivity)