cmake_minimum_required(VERSION 3.10)

project(MSIRRT_Planner_Planner)

if(DEFINED CMAKE_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "-march=native -mtune=native -O3 -no-pie -fno-pie -flto=auto -fwhole-program -fomit-frame-pointer -fdata-sections -ffunction-sections -Wl,--gc-sections")
        set(CMAKE_C_FLAGS "-march=native -mtune=native -O3 -no-pie -fno-pie -flto=auto -fwhole-program -fomit-frame-pointer -fdata-sections -ffunction-sections -Wl,--gc-sections")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "-O0 -g3 -fno-omit-frame-pointer -Wall -Wextra -fsanitize=address -fsanitize=undefined  -fsanitize=leak ")
        set(CMAKE_C_FLAGS "-O0 -g3 -fno-omit-frame-pointer -Wall -Wextra -fsanitize=address -fsanitize=undefined  -fsanitize=leak ")
    endif()
else()
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "-march=native -mtune=native -O3 -no-pie -fno-pie -flto=auto -fwhole-program -fomit-frame-pointer -fdata-sections -ffunction-sections -Wl,--gc-sections")
    set(CMAKE_C_FLAGS "-march=native -mtune=native -O3 -no-pie -fno-pie -flto=auto -fwhole-program -fomit-frame-pointer -fdata-sections -ffunction-sections -Wl,--gc-sections")
endif()



message(STATUS "CMAKE_BUILD_TYPE is set to '${CMAKE_BUILD_TYPE}'")
message(STATUS "CMAKE_CXX_FLAGS is set to '${CMAKE_CXX_FLAGS}'")
message(STATUS "CMAKE_C_FLAGS is set to '${CMAKE_C_FLAGS}'")


# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find RapidJSON
find_package(RapidJSON REQUIRED)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${Eigen3_INCLUDE_DIRS})
find_package(RapidJSON REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../rapidjson/)

# Include directories
include_directories(include)

add_subdirectory("../STRRT_Planner/src/config_read_writer" ./config_read_writer)

add_subdirectory("../STRRT_Planner/src/CollisionManager" ./CollisionManager)

# Add executable
add_library(MSIRRT_Planner_lib STATIC  "./src/PlannerConnect.cpp" "./src/RepairTypes.cpp" "./src/Tree.cpp" "./src/TreeRepair.cpp" "./src/TubeAblation.cpp" "./src/Vertex.cpp")
target_link_libraries(MSIRRT_Planner_lib PUBLIC config_read_writer CollisionManager)  
target_include_directories(MSIRRT_Planner_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_executable(MSIRRT_Planner "src/main.cpp")
add_executable(MSIRRT_RepairSequence "src/online_main.cpp")


# Link libraries
target_link_libraries(MSIRRT_Planner PUBLIC MSIRRT_Planner_lib urdf orocos-kdl Eigen3::Eigen config_read_writer CollisionManager   pthread )
target_link_libraries(MSIRRT_RepairSequence PUBLIC MSIRRT_Planner_lib CollisionManager config_read_writer urdf orocos-kdl Eigen3::Eigen pthread)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_repair_types "tests/test_repair_types.cpp" "src/RepairTypes.cpp")
    target_compile_options(test_repair_types PRIVATE -UNDEBUG)
    target_include_directories(test_repair_types PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_test(NAME repair_types COMMAND test_repair_types)

    add_executable(test_tree_repair "tests/test_tree_repair.cpp" "src/RepairTypes.cpp" "src/Tree.cpp" "src/TreeRepair.cpp" "src/Vertex.cpp")
    target_compile_options(test_tree_repair PRIVATE -UNDEBUG)
    target_include_directories(test_tree_repair PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(test_tree_repair PRIVATE Eigen3::Eigen)
    add_test(NAME tree_repair COMMAND test_tree_repair)

    add_executable(test_tree_repair_engine "tests/test_tree_repair_engine.cpp" "src/RepairTypes.cpp" "src/Tree.cpp" "src/TreeRepair.cpp" "src/Vertex.cpp")
    target_compile_options(test_tree_repair_engine PRIVATE -UNDEBUG)
    target_include_directories(test_tree_repair_engine PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(test_tree_repair_engine PRIVATE Eigen3::Eigen)
    add_test(NAME tree_repair_engine COMMAND test_tree_repair_engine)

    add_executable(test_prediction_update "tests/test_prediction_update.cpp")
    target_compile_options(test_prediction_update PRIVATE -UNDEBUG)
    target_link_libraries(
        test_prediction_update
        PRIVATE MSIRRT_Planner_lib CollisionManager config_read_writer urdf orocos-kdl Eigen3::Eigen pthread
    )
    add_test(
        NAME prediction_update
        COMMAND test_prediction_update
            "${CMAKE_CURRENT_SOURCE_DIR}/../examples/one_sphere/scene_task.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/../examples/twenty_spheres/scene_task.json"
    )
    set_tests_properties(
        prediction_update
        PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=/opt/SIRRT_dependencies/release/lib:/usr/local/lib:/opt/ros/jazzy/lib;AMENT_PREFIX_PATH=/opt/ros/jazzy"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )

    add_executable(test_safe_interval_equivalence "tests/test_safe_interval_equivalence.cpp")
    target_compile_options(test_safe_interval_equivalence PRIVATE -UNDEBUG)
    target_link_libraries(
        test_safe_interval_equivalence
        PRIVATE MSIRRT_Planner_lib CollisionManager config_read_writer urdf orocos-kdl Eigen3::Eigen pthread
    )
    add_test(
        NAME safe_interval_equivalence
        COMMAND test_safe_interval_equivalence
            "${CMAKE_CURRENT_SOURCE_DIR}/../examples/one_sphere/scene_task.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/../examples/twenty_spheres/scene_task.json"
    )
    set_tests_properties(
        safe_interval_equivalence
        PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=/opt/SIRRT_dependencies/release/lib:/usr/local/lib:/opt/ros/jazzy/lib;AMENT_PREFIX_PATH=/opt/ros/jazzy"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )

    add_executable(test_rolling_reroot "tests/test_rolling_reroot.cpp")
    target_compile_options(test_rolling_reroot PRIVATE -UNDEBUG)
    target_link_libraries(
        test_rolling_reroot
        PRIVATE MSIRRT_Planner_lib CollisionManager config_read_writer urdf orocos-kdl Eigen3::Eigen pthread
    )
    add_test(
        NAME rolling_reroot
        COMMAND test_rolling_reroot "${CMAKE_CURRENT_SOURCE_DIR}/../examples/one_sphere/scene_task.json"
    )
    set_tests_properties(
        rolling_reroot
        PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=/opt/SIRRT_dependencies/release/lib:/usr/local/lib:/opt/ros/jazzy/lib;AMENT_PREFIX_PATH=/opt/ros/jazzy;MSIRRT_MAX_PLANNING_TIME=2"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )

    add_executable(test_tube_ablation "tests/test_tube_ablation.cpp")
    target_compile_options(test_tube_ablation PRIVATE -UNDEBUG)
    target_link_libraries(
        test_tube_ablation
        PRIVATE MSIRRT_Planner_lib CollisionManager config_read_writer urdf orocos-kdl Eigen3::Eigen pthread
    )
    add_test(
        NAME tube_ablation
        COMMAND test_tube_ablation
            "${CMAKE_CURRENT_SOURCE_DIR}/../examples/twenty_spheres/scene_task.json"
    )
    set_tests_properties(
        tube_ablation
        PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=/opt/SIRRT_dependencies/release/lib:/usr/local/lib:/opt/ros/jazzy/lib;AMENT_PREFIX_PATH=/opt/ros/jazzy"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
endif()
