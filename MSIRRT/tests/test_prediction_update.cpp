#include "PlannerConnect.hpp"
#include "config_read_writer/SphereObstacleJsonInfo.hpp"
#include "config_read_writer/config_read.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace MDP::MSIRRT
{

struct PlannerConnectTestAccess
{
    static std::vector<Vertex *> set_goal_parent_without_start_root(PlannerConnect &planner)
    {
        planner.root_node = nullptr;
        planner.current_tree = planner.goal_tree;
        auto coordinates = planner.goal_coords;
        auto safe_intervals = planner.goal_safe_intervals;
        return planner.set_parent(coordinates, safe_intervals);
    }

    static Vertex::VertexCoordType sample_configuration(PlannerConnect &planner)
    {
        return planner.get_random_configuration();
    }

    static bool is_directly_reachable(
        PlannerConnect &planner,
        const Vertex::VertexCoordType &coordinates)
    {
        const Vertex::VertexCoordType start(planner.scene_task.start_configuration.data());
        const double available_distance =
            static_cast<double>(planner.planning_horizon_frame - planner.planning_start_frame) *
            planner.vmax / static_cast<double>(planner.scene_task.fps);
        return (coordinates - start).norm() +
                (coordinates - planner.goal_coords).norm() <=
            available_distance + 1e-9;
    }

    static bool is_in_joint_limits(
        PlannerConnect &planner,
        const Vertex::VertexCoordType &coordinates)
    {
        return planner.is_coords_in_limits(coordinates);
    }

    static Vertex::VertexCoordType start_configuration(PlannerConnect &planner)
    {
        return Vertex::VertexCoordType(planner.scene_task.start_configuration.data());
    }
};

} // namespace MDP::MSIRRT

namespace
{

using ObstacleCoordinatesReturn = decltype(
    std::declval<const MDP::ObstacleJsonInfo &>().get_coordinates());
static_assert(
    std::is_lvalue_reference_v<ObstacleCoordinatesReturn>,
    "obstacle coordinates must be exposed without copying");
static_assert(
    std::is_const_v<std::remove_reference_t<ObstacleCoordinatesReturn>>,
    "obstacle coordinates must remain immutable to callers");
static_assert(
    std::is_constructible_v<
        MDP::SphereObstaclesFCL,
        const MDP::SphereObstacleJsonInfo &>,
    "sphere collision obstacles must accept immutable JSON obstacles");
static_assert(
    std::is_constructible_v<
        MDP::CubeObstaclesFCL,
        const MDP::CubeObstacleJsonInfo &>,
    "box collision obstacles must accept immutable JSON obstacles");

std::vector<std::vector<float>> sphere_coordinates(
    const MDP::SphereObstacleJsonInfo &sphere)
{
    std::vector<std::vector<float>> coordinates;
    for (const auto &position : sphere.get_coordinates())
    {
        coordinates.push_back({
            position.x,
            position.y,
            position.z,
            position.quat_x,
            position.quat_y,
            position.quat_z,
            position.quat_w});
    }
    return coordinates;
}

MDP::ConfigReader::SceneTask with_inflation(
    const MDP::ConfigReader::SceneTask &source,
    std::size_t first_frame,
    std::size_t last_frame)
{
    auto updated = source;
    for (std::size_t obstacle_id = 0; obstacle_id < updated.obstacles.size(); ++obstacle_id)
    {
        const auto &obstacle = updated.obstacles[obstacle_id];
        if (obstacle->get_type() != "dynamic_sphere")
        {
            continue;
        }
        const auto *sphere = static_cast<const MDP::SphereObstacleJsonInfo *>(obstacle.get());
        std::vector<float> uncertainty(sphere->get_coordinates().size(), 0.0F);
        for (std::size_t frame = first_frame; frame <= std::min(last_frame, uncertainty.size() - 1); ++frame)
        {
            uncertainty[frame] = 0.02F;
        }
        updated.obstacles[obstacle_id] = std::make_shared<MDP::SphereObstacleJsonInfo>(
            sphere->get_name(),
            sphere->get_type(),
            sphere_coordinates(*sphere),
            static_cast<float>(updated.fps),
            sphere->get_radius(),
            false,
            uncertainty);
        return updated;
    }
    throw std::runtime_error("test scene has no dynamic sphere");
}

MDP::ConfigReader::SceneTask with_changed_radius(
    const MDP::ConfigReader::SceneTask &source)
{
    auto updated = source;
    for (std::size_t obstacle_id = 0; obstacle_id < updated.obstacles.size(); ++obstacle_id)
    {
        const auto &obstacle = updated.obstacles[obstacle_id];
        if (obstacle->get_obstacle_type() != MDP::ObstacleJsonInfo::ObstacleType::SPHERE)
        {
            continue;
        }
        const auto *sphere = static_cast<const MDP::SphereObstacleJsonInfo *>(obstacle.get());
        updated.obstacles[obstacle_id] = std::make_shared<MDP::SphereObstacleJsonInfo>(
            sphere->get_name(),
            sphere->get_type(),
            sphere_coordinates(*sphere),
            static_cast<float>(updated.fps),
            sphere->get_radius() + 0.01F,
            sphere->get_is_static(),
            sphere->get_uncertainty_radii());
        return updated;
    }
    throw std::runtime_error("test scene has no sphere");
}

MDP::ConfigReader::SceneTask with_first_sphere_constant(
    const MDP::ConfigReader::SceneTask &source)
{
    auto updated = source;
    for (std::size_t obstacle_id = 0; obstacle_id < updated.obstacles.size(); ++obstacle_id)
    {
        const auto &obstacle = updated.obstacles[obstacle_id];
        if (obstacle->get_type() != "dynamic_sphere")
        {
            continue;
        }
        const auto *sphere = static_cast<const MDP::SphereObstacleJsonInfo *>(obstacle.get());
        auto coordinates = sphere_coordinates(*sphere);
        std::fill(coordinates.begin(), coordinates.end(), coordinates.front());
        updated.obstacles[obstacle_id] = std::make_shared<MDP::SphereObstacleJsonInfo>(
            sphere->get_name(),
            sphere->get_type(),
            coordinates,
            static_cast<float>(updated.fps),
            sphere->get_radius(),
            false,
            sphere->get_uncertainty_radii());
        return updated;
    }
    throw std::runtime_error("test scene has no dynamic sphere");
}

MDP::ConfigReader::SceneTask with_first_sphere_excursion(
    const MDP::ConfigReader::SceneTask &source,
    std::size_t first_frame,
    std::size_t last_frame)
{
    auto updated = source;
    for (std::size_t obstacle_id = 0; obstacle_id < updated.obstacles.size(); ++obstacle_id)
    {
        const auto &obstacle = updated.obstacles[obstacle_id];
        if (obstacle->get_type() != "dynamic_sphere")
        {
            continue;
        }
        const auto *sphere = static_cast<const MDP::SphereObstacleJsonInfo *>(obstacle.get());
        auto coordinates = sphere_coordinates(*sphere);
        for (std::size_t frame = first_frame; frame <= std::min(last_frame, coordinates.size() - 1); ++frame)
        {
            coordinates[frame][0] += 0.001F * static_cast<float>(frame - first_frame + 1);
        }
        updated.obstacles[obstacle_id] = std::make_shared<MDP::SphereObstacleJsonInfo>(
            sphere->get_name(),
            sphere->get_type(),
            coordinates,
            static_cast<float>(updated.fps),
            sphere->get_radius(),
            false,
            sphere->get_uncertainty_radii());
        return updated;
    }
    throw std::runtime_error("test scene has no dynamic sphere");
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: test_prediction_update <scene.json> <dense-scene.json>\n";
        return 2;
    }

    MDP::ConfigReader reader(argv[1]);
    auto initial = reader.get_scene_task();
    auto updated = with_inflation(initial, 10, 20);

    MDP::CollisionManager locally_updated(initial);
    const auto local_report = locally_updated.update_scene(updated, {{10, 20}});
    assert(!local_report.full_rebuild_fallback);
    assert(local_report.frames_rebuilt == 11);
    MDP::CollisionManager fully_rebuilt(updated);
    auto midpoint = initial.start_configuration;
    for (std::size_t joint = 0; joint < midpoint.size(); ++joint)
    {
        midpoint[joint] = 0.5 * (initial.start_configuration[joint] + initial.end_configuration[joint]);
    }
    for (const auto &configuration : {
             initial.start_configuration,
             midpoint,
             initial.end_configuration})
    {
        for (const int frame : {0, 9, 10, 15, 20, 21, 599})
        {
            assert(
                locally_updated.check_collision_frame_no_wrapper(configuration, frame) ==
                fully_rebuilt.check_collision_frame_no_wrapper(configuration, frame));
        }
        assert(
            locally_updated.get_safe_intervals(configuration) ==
            fully_rebuilt.get_safe_intervals(configuration));
    }

    auto incompatible = initial;
    incompatible.frame_count -= 1;
    const auto fallback_report = locally_updated.update_scene(incompatible, {{10, 20}});
    assert(fallback_report.full_rebuild_fallback);
    assert(locally_updated.update_scene(with_changed_radius(initial), {{10, 20}}).full_rebuild_fallback);
    auto missing_obstacle = initial;
    missing_obstacle.obstacles.pop_back();
    assert(locally_updated.update_scene(missing_obstacle, {{10, 20}}).full_rebuild_fallback);
    auto changed_robot = initial;
    changed_robot.robot_urdf_path += ".different";
    assert(locally_updated.update_scene(changed_robot, {{10, 20}}).full_rebuild_fallback);

    MDP::ConfigReader dense_reader(argv[2]);
    const auto dense_initial = with_first_sphere_constant(dense_reader.get_scene_task());
    const auto dense_updated = with_first_sphere_excursion(dense_initial, 105, 134);
    MDP::CollisionManager dense_locally_updated(dense_initial);
    const auto dense_report = dense_locally_updated.update_scene(dense_updated, {{105, 135}});
    assert(!dense_report.full_rebuild_fallback);
    assert(dense_report.frames_rebuilt == 31);
    assert(dense_report.obstacles_reindexed == 1);
    MDP::CollisionManager dense_fully_rebuilt(dense_updated);
    for (const int frame : {0, 104, 105, 120, 134, 135, 599})
    {
        assert(
            dense_locally_updated.check_collision_frame_no_wrapper(initial.start_configuration, frame) ==
            dense_fully_rebuilt.check_collision_frame_no_wrapper(initial.start_configuration, frame));
    }
    assert(
        dense_locally_updated.get_safe_intervals(initial.start_configuration) ==
        dense_fully_rebuilt.get_safe_intervals(initial.start_configuration));

    MDP::CollisionManager dense_unhinted_full_update(dense_initial);
    const auto unhinted_full_report = dense_unhinted_full_update.update_scene(
        dense_updated, {});
    assert(!unhinted_full_report.full_rebuild_fallback);
    assert(unhinted_full_report.frames_rebuilt == 30);
    assert(unhinted_full_report.obstacles_reindexed == 1);
    for (const auto &configuration : {
             dense_initial.start_configuration,
             dense_initial.end_configuration})
    {
        for (const int frame : {0, 104, 105, 120, 134, 135, 599})
        {
            assert(
                dense_unhinted_full_update.check_collision_frame_no_wrapper(
                    configuration, frame) ==
                dense_fully_rebuilt.check_collision_frame_no_wrapper(
                    configuration, frame));
        }
        assert(
            dense_unhinted_full_update.get_safe_intervals(configuration) ==
            dense_fully_rebuilt.get_safe_intervals(configuration));
    }

    MDP::CollisionManager dense_horizon_bounded(dense_initial);
    const auto bounded_report = dense_horizon_bounded.update_scene(
        dense_updated, {{105, 120}}, 120);
    assert(!bounded_report.full_rebuild_fallback);
    assert(bounded_report.frames_rebuilt == 16);
    assert(bounded_report.obstacles_reindexed == 1);
    for (const auto &configuration : {
             dense_initial.start_configuration,
             dense_initial.end_configuration})
    {
        for (const int frame : {0, 104, 105, 110, 120})
        {
            assert(
                dense_horizon_bounded.check_collision_frame_no_wrapper(
                    configuration, frame) ==
                dense_fully_rebuilt.check_collision_frame_no_wrapper(
                    configuration, frame));
        }
        assert(
            dense_horizon_bounded.get_safe_intervals(configuration, 120) ==
            dense_fully_rebuilt.get_safe_intervals(configuration, 120));
    }

    const auto expanded_report = dense_horizon_bounded.update_scene(
        dense_updated, {{121, 135}}, 135);
    assert(!expanded_report.full_rebuild_fallback);
    assert(expanded_report.frames_rebuilt == 15);
    assert(expanded_report.obstacles_reindexed == 1);
    for (const auto &configuration : {
             dense_initial.start_configuration,
             dense_initial.end_configuration})
    {
        for (const int frame : {0, 104, 105, 120, 121, 134, 135})
        {
            assert(
                dense_horizon_bounded.check_collision_frame_no_wrapper(
                    configuration, frame) ==
                dense_fully_rebuilt.check_collision_frame_no_wrapper(
                    configuration, frame));
        }
        assert(
            dense_horizon_bounded.get_safe_intervals(configuration, 135) ==
            dense_fully_rebuilt.get_safe_intervals(configuration, 135));
    }

    MDP::CollisionManager dense_unhinted_bounded_update(dense_initial);
    const auto unhinted_bounded_report = dense_unhinted_bounded_update.update_scene(
        dense_updated, {}, 120);
    assert(!unhinted_bounded_report.full_rebuild_fallback);
    assert(unhinted_bounded_report.frames_rebuilt == 16);
    assert(unhinted_bounded_report.obstacles_reindexed == 1);
    for (const auto &configuration : {
             dense_initial.start_configuration,
             dense_initial.end_configuration})
    {
        for (const int frame : {0, 104, 105, 110, 120})
        {
            assert(
                dense_unhinted_bounded_update.check_collision_frame_no_wrapper(
                    configuration, frame) ==
                dense_fully_rebuilt.check_collision_frame_no_wrapper(
                    configuration, frame));
        }
        assert(
            dense_unhinted_bounded_update.get_safe_intervals(configuration, 120) ==
            dense_fully_rebuilt.get_safe_intervals(configuration, 120));
    }

    MDP::CollisionManager dense_deferred_future_update(dense_initial);
    const auto deferred_report = dense_deferred_future_update.update_scene(
        dense_updated, {}, 100);
    assert(!deferred_report.full_rebuild_fallback);
    assert(deferred_report.frames_rebuilt == 0);
    assert(deferred_report.obstacles_reindexed == 0);
    bool query_past_deferred_horizon_rejected = false;
    try
    {
        static_cast<void>(dense_deferred_future_update.check_collision_frame_no_wrapper(
            dense_initial.start_configuration, 101));
    }
    catch (const std::out_of_range &)
    {
        query_past_deferred_horizon_rejected = true;
    }
    bool safe_intervals_past_deferred_horizon_rejected = false;
    try
    {
        static_cast<void>(dense_deferred_future_update.get_safe_intervals(
            dense_initial.start_configuration, 101));
    }
    catch (const std::out_of_range &)
    {
        safe_intervals_past_deferred_horizon_rejected = true;
    }
    bool distance_past_deferred_horizon_rejected = false;
    try
    {
        bool is_collision = false;
        static_cast<void>(dense_deferred_future_update.get_distances(
            dense_initial.start_configuration,
            101.5F / static_cast<float>(dense_initial.fps),
            is_collision));
    }
    catch (const std::out_of_range &)
    {
        distance_past_deferred_horizon_rejected = true;
    }

    const auto deferred_expansion_report = dense_deferred_future_update.update_scene(
        dense_updated, {}, 135);
    assert(!deferred_expansion_report.full_rebuild_fallback);
    assert(deferred_expansion_report.frames_rebuilt == 35);
    assert(deferred_expansion_report.obstacles_reindexed == 1);
    assert(query_past_deferred_horizon_rejected);
    assert(safe_intervals_past_deferred_horizon_rejected);
    assert(distance_past_deferred_horizon_rejected);
    for (const auto &configuration : {
             dense_initial.start_configuration,
             dense_initial.end_configuration})
    {
        for (const int frame : {0, 100, 101, 105, 120, 134, 135})
        {
            assert(
                dense_deferred_future_update.check_collision_frame_no_wrapper(
                    configuration, frame) ==
                dense_fully_rebuilt.check_collision_frame_no_wrapper(
                    configuration, frame));
        }
        assert(
            dense_deferred_future_update.get_safe_intervals(configuration, 135) ==
            dense_fully_rebuilt.get_safe_intervals(configuration, 135));
    }

    setenv("MSIRRT_REUSE_STORED_PREVIOUS_INTERVALS", "1", 1);
    MDP::MSIRRT::PlannerConnect planner(initial, 42);
    const auto first = planner.update_prediction(updated);
    assert(first.prediction_version == 1);
    assert(first.changed_windows.size() == 1);
    assert(first.changed_windows[0] == MDP::MSIRRT::FrameRange(10, 20));
    assert(first.start_tree.candidate_vertices > 0);
    assert(first.goal_tree.candidate_vertices > 0);
    assert(first.used_stored_previous_intervals);
    assert(first.invariants_hold);
    assert(first.tube_update_seconds >= 0.0);
    assert(first.safe_interval_update_seconds >= first.collision_structure_update_seconds);
    assert(first.tree_repair_seconds >= 0.0);
    assert(first.repair_seconds + 1e-9 >= first.tree_repair_seconds);

    const auto second = planner.update_prediction(updated);
    assert(second.prediction_version == 2);
    assert(second.changed_windows.empty());
    assert(second.collision_structure_update_seconds == 0.0);
    assert(second.tube_update_seconds >= 0.0);
    assert(second.safe_interval_update_seconds >= 0.0);
    assert(second.tree_repair_seconds >= 0.0);
    assert(second.invariants_hold);
    unsetenv("MSIRRT_REUSE_STORED_PREVIOUS_INTERVALS");

    MDP::MSIRRT::PlannerConnect snapshot_planner(initial, 43);
    const auto snapshot = snapshot_planner.update_prediction(updated);
    assert(!snapshot.used_stored_previous_intervals);
    assert(snapshot.invariants_hold);

    auto limited = initial;
    limited.reliable_until_frame = 30;
    auto changed_after_horizon = with_inflation(limited, 40, 50);
    MDP::MSIRRT::PlannerConnect limited_planner(limited, 42);
    const auto ignored = limited_planner.update_prediction(changed_after_horizon);
    assert(ignored.changed_windows.empty());
    assert(ignored.collision_structure_update_seconds == 0.0);
    assert(ignored.invariants_hold);

    MDP::MSIRRT::PlannerConnect solved_planner(initial, 42);
    assert(solved_planner.solve());
    const auto original_path = solved_planner.get_final_path();
    assert(!original_path.empty());
    const auto original_finish_id = original_path.back()->vertex_id;
    auto changed_away_from_solution = with_inflation(initial, 100, 110);
    const auto preserved = solved_planner.update_prediction(changed_away_from_solution);
    assert(!preserved.changed_windows.empty());
    assert(preserved.previous_solution_preserved);
    assert(!preserved.previous_solution_invalidated);
    assert(solved_planner.has_valid_solution());
    solved_planner.set_max_planning_time(0.0001);
    assert(solved_planner.solve());
    const auto preserved_path = solved_planner.get_final_path();
    assert(!preserved_path.empty());
    assert(preserved_path.back()->vertex_id == original_finish_id);

    setenv("MSIRRT_CONDITIONAL_SAMPLING", "1", 1);
    setenv("MSIRRT_CONDITIONAL_EXPLORATION_RATE", "0", 1);
    MDP::MSIRRT::PlannerConnect conditional_planner(initial, 123);
    conditional_planner.set_max_planning_time(0.01);
    static_cast<void>(conditional_planner.solve());
    assert(
        conditional_planner.get_conditionally_rejected_samples() ==
        conditional_planner.get_conditionally_rejected_kinematic_samples() +
            conditional_planner.get_conditionally_rejected_temporal_samples());

    setenv("MSIRRT_COOPERATIVE_DEADLINE_CHECKS", "1", 1);
    MDP::MSIRRT::PlannerConnect deadline_planner(dense_initial, 126);
    assert(deadline_planner.get_cooperative_deadline_checks_enabled());
    deadline_planner.set_max_planning_time(0.005);
    const auto deadline_start = std::chrono::steady_clock::now();
    static_cast<void>(deadline_planner.solve());
    const double deadline_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - deadline_start).count();
    assert(deadline_elapsed < 0.020);
    assert(deadline_planner.get_cooperative_deadline_check_triggers() > 0);
    unsetenv("MSIRRT_COOPERATIVE_DEADLINE_CHECKS");

    auto reachable_scene = initial;
    reachable_scene.reliable_until_frame = 60;
    setenv("MSIRRT_REACHABLE_ELLIPSOID_SAMPLING", "1", 1);
    MDP::MSIRRT::PlannerConnect reachable_sampler(reachable_scene, 127);
    assert(reachable_sampler.get_reachable_ellipsoid_sampling_enabled());
    for (int sample = 0; sample < 128; ++sample)
    {
        const auto coordinates =
            MDP::MSIRRT::PlannerConnectTestAccess::sample_configuration(
                reachable_sampler);
        assert(MDP::MSIRRT::PlannerConnectTestAccess::is_directly_reachable(
            reachable_sampler, coordinates));
        assert(MDP::MSIRRT::PlannerConnectTestAccess::is_in_joint_limits(
            reachable_sampler, coordinates));
    }
    assert(reachable_sampler.get_reachable_ellipsoid_samples() == 128);
    unsetenv("MSIRRT_REACHABLE_ELLIPSOID_SAMPLING");

    MDP::MSIRRT::PlannerConnect default_sampler(reachable_scene, 128);
    setenv("MSIRRT_REACHABLE_ELLIPSOID_SAMPLING", "0", 1);
    MDP::MSIRRT::PlannerConnect explicitly_disabled_sampler(reachable_scene, 128);
    assert(!default_sampler.get_reachable_ellipsoid_sampling_enabled());
    assert(!explicitly_disabled_sampler.get_reachable_ellipsoid_sampling_enabled());
    for (int sample = 0; sample < 32; ++sample)
    {
        const auto default_coordinates =
            MDP::MSIRRT::PlannerConnectTestAccess::sample_configuration(
                default_sampler);
        const auto disabled_coordinates =
            MDP::MSIRRT::PlannerConnectTestAccess::sample_configuration(
                explicitly_disabled_sampler);
        assert(default_coordinates == disabled_coordinates);
    }
    assert(default_sampler.get_reachable_ellipsoid_samples() == 0);
    assert(explicitly_disabled_sampler.get_reachable_ellipsoid_samples() == 0);
    unsetenv("MSIRRT_REACHABLE_ELLIPSOID_SAMPLING");

    setenv("MSIRRT_REJECTION_FEEDBACK", "1", 1);
    setenv("MSIRRT_REJECTION_WINDOW", "1", 1);
    setenv("MSIRRT_REJECTION_THRESHOLD", "0", 1);
    setenv("MSIRRT_REJECTION_EXPLORATION_BOOST", "1", 1);
    MDP::MSIRRT::PlannerConnect feedback_planner(initial, 124);
    feedback_planner.set_max_planning_time(0.01);
    static_cast<void>(feedback_planner.solve());
    assert(feedback_planner.get_rejection_feedback_enabled());
    assert(feedback_planner.get_conditional_rejection_feedback_triggers() > 0);
    assert(
        std::abs(feedback_planner.get_conditional_exploration_effective_rate() - 1.0) <
        1e-12);

    setenv("MSIRRT_ADAPTIVE_REPAIR_POLICY", "legacy", 1);
    MDP::MSIRRT::PlannerConnect legacy_feedback_planner(initial, 125);
    assert(!legacy_feedback_planner.get_rejection_feedback_enabled());
    unsetenv("MSIRRT_ADAPTIVE_REPAIR_POLICY");
    unsetenv("MSIRRT_REJECTION_FEEDBACK");
    unsetenv("MSIRRT_REJECTION_WINDOW");
    unsetenv("MSIRRT_REJECTION_THRESHOLD");
    unsetenv("MSIRRT_REJECTION_EXPLORATION_BOOST");

    MDP::MSIRRT::PlannerConnect missing_root_planner(initial, 456);
    assert(
        MDP::MSIRRT::PlannerConnectTestAccess::set_goal_parent_without_start_root(
            missing_root_planner).empty());
    assert(!missing_root_planner.solve());

    std::cout << "prediction update integration test passed\n";
    return 0;
}
