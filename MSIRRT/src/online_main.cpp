#include "PlannerConnect.hpp"
#include "TubeAblation.hpp"
#include "CollisionManager/CollisionManager.hpp"
#include "config_read_writer/SphereObstacleJsonInfo.hpp"
#include "config_read_writer/config_read.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

rapidjson::Document read_json(const std::filesystem::path &path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("cannot open JSON file: " + path.string());
    }
    rapidjson::IStreamWrapper wrapper(input);
    rapidjson::Document document;
    document.ParseStream(wrapper);
    if (document.HasParseError() || !document.IsObject())
    {
        throw std::runtime_error("invalid JSON object: " + path.string());
    }
    return document;
}

std::string prediction_status_from_document(const rapidjson::Document &document)
{
    const auto experiment_it = document.FindMember("_uncertainty_experiment");
    if (experiment_it == document.MemberEnd() || !experiment_it->value.IsObject())
    {
        return "unspecified";
    }
    const auto status_it = experiment_it->value.FindMember("prediction_status");
    return status_it != experiment_it->value.MemberEnd() && status_it->value.IsString()
        ? status_it->value.GetString()
        : "unspecified";
}

int positive_environment_integer(const char *name, int fallback)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr)
    {
        return fallback;
    }
    const int value = std::atoi(raw);
    return value > 0 ? value : fallback;
}

int nonnegative_environment_integer(const char *name, int fallback)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr)
    {
        return fallback;
    }
    const int value = std::atoi(raw);
    return value >= 0 ? value : fallback;
}

double bounded_environment_double(
    const char *name,
    double fallback,
    double minimum,
    double maximum)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr)
    {
        return fallback;
    }
    std::size_t consumed = 0;
    const double value = std::stod(raw, &consumed);
    if (raw[consumed] != '\0' || !std::isfinite(value) || value < minimum ||
        value > maximum)
    {
        throw std::invalid_argument(std::string(name) + " is outside its valid range");
    }
    return value;
}

bool environment_flag(const char *name, bool fallback)
{
    const char *raw = std::getenv(name);
    return raw == nullptr ? fallback : std::atoi(raw) != 0;
}

double required_manifest_config_double(
    const rapidjson::Document &manifest,
    const char *name)
{
    if (!manifest.HasMember("config") || !manifest["config"].IsObject())
    {
        throw std::invalid_argument("fixed-tube ablation requires manifest.config");
    }
    const auto member = manifest["config"].FindMember(name);
    if (member == manifest["config"].MemberEnd() || !member->value.IsNumber())
    {
        throw std::invalid_argument(
            std::string("fixed-tube ablation requires numeric manifest.config.") + name);
    }
    const double value = member->value.GetDouble();
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::invalid_argument(
            std::string("fixed-tube manifest configuration is invalid: ") + name);
    }
    return value;
}

void accumulate_repair_report(
    MDP::MSIRRT::RepairReport &target,
    const MDP::MSIRRT::RepairReport &addition)
{
    MDP::MSIRRT::accumulate_sequential_repair_report(target, addition);
}

void accumulate_bidirectional_repair(
    MDP::MSIRRT::BidirectionalRepairReport &target,
    const MDP::MSIRRT::BidirectionalRepairReport &addition)
{
    target.prediction_version = addition.prediction_version;
    target.changed_windows.insert(
        target.changed_windows.end(),
        addition.changed_windows.begin(),
        addition.changed_windows.end());
    target.changed_windows = MDP::MSIRRT::normalize_frame_ranges(target.changed_windows);
    accumulate_repair_report(target.start_tree, addition.start_tree);
    accumulate_repair_report(target.goal_tree, addition.goal_tree);
    target.collision_structure_update_seconds += addition.collision_structure_update_seconds;
    target.collision_frames_rebuilt += addition.collision_frames_rebuilt;
    target.collision_obstacles_reindexed += addition.collision_obstacles_reindexed;
    target.collision_full_rebuild_fallback =
        target.collision_full_rebuild_fallback || addition.collision_full_rebuild_fallback;
    target.tube_update_seconds += addition.tube_update_seconds;
    target.safe_interval_update_seconds += addition.safe_interval_update_seconds;
    target.tree_repair_seconds += addition.tree_repair_seconds;
    target.repair_seconds += addition.repair_seconds;
    target.used_stored_previous_intervals =
        target.used_stored_previous_intervals || addition.used_stored_previous_intervals;
    target.previous_solution_invalidated =
        target.previous_solution_invalidated || addition.previous_solution_invalidated;
    target.previous_solution_preserved =
        target.previous_solution_preserved || addition.previous_solution_preserved;
    target.invariants_hold = target.invariants_hold && addition.invariants_hold;
}

rapidjson::Value frame_ranges_json(
    const std::vector<MDP::MSIRRT::FrameRange> &ranges,
    rapidjson::Document::AllocatorType &allocator)
{
    rapidjson::Value result(rapidjson::kArrayType);
    for (const auto &range : ranges)
    {
        rapidjson::Value value(rapidjson::kArrayType);
        value.PushBack(range.first, allocator);
        value.PushBack(range.last, allocator);
        result.PushBack(value, allocator);
    }
    return result;
}

void add_nullable_frame(
    rapidjson::Value &object,
    const char *name,
    int frame,
    rapidjson::Document::AllocatorType &allocator)
{
    rapidjson::Value key(name, allocator);
    if (frame >= 0)
    {
        object.AddMember(key, frame, allocator);
    }
    else
    {
        object.AddMember(key, rapidjson::Value().SetNull(), allocator);
    }
}

rapidjson::Value repair_report_json(
    const MDP::MSIRRT::RepairReport &report,
    rapidjson::Document::AllocatorType &allocator)
{
    rapidjson::Value value(rapidjson::kObjectType);
    value.AddMember("active_vertices_before", static_cast<uint64_t>(report.active_vertices_before), allocator);
    value.AddMember("candidate_vertices", static_cast<uint64_t>(report.candidate_vertices), allocator);
    value.AddMember("candidate_edges", static_cast<uint64_t>(report.candidate_edges), allocator);
    value.AddMember("revalidated_vertices", static_cast<uint64_t>(report.revalidated_vertices), allocator);
    value.AddMember("revalidated_edges", static_cast<uint64_t>(report.revalidated_edges), allocator);
    value.AddMember("reused_vertices", static_cast<uint64_t>(report.reused_vertices), allocator);
    value.AddMember("invalidated_vertices", static_cast<uint64_t>(report.invalidated_vertices), allocator);
    value.AddMember("reconnected_vertices", static_cast<uint64_t>(report.reconnected_vertices), allocator);
    value.AddMember("intervals_unchanged", static_cast<uint64_t>(report.intervals_unchanged), allocator);
    value.AddMember("intervals_shrunk", static_cast<uint64_t>(report.intervals_shrunk), allocator);
    value.AddMember("intervals_expanded", static_cast<uint64_t>(report.intervals_expanded), allocator);
    value.AddMember("intervals_split", static_cast<uint64_t>(report.intervals_split), allocator);
    value.AddMember("intervals_shifted", static_cast<uint64_t>(report.intervals_shifted), allocator);
    value.AddMember("intervals_merged", static_cast<uint64_t>(report.intervals_merged), allocator);
    value.AddMember("intervals_deleted", static_cast<uint64_t>(report.intervals_deleted), allocator);
    value.AddMember("vertex_reuse_rate", report.vertex_reuse_rate(), allocator);
    value.AddMember("invariants_hold", report.invariants_hold, allocator);
    value.AddMember(
        "invariant_error",
        rapidjson::Value(report.invariant_error.c_str(), allocator),
        allocator);
    return value;
}

rapidjson::Value final_path_json(
    const std::vector<MDP::MSIRRT::Vertex *> &path,
    rapidjson::Document::AllocatorType &allocator)
{
    rapidjson::Value result(rapidjson::kArrayType);
    for (const MDP::MSIRRT::Vertex *vertex : path)
    {
        rapidjson::Value item(rapidjson::kObjectType);
        rapidjson::Value coordinates(rapidjson::kArrayType);
        for (int joint = 0; joint < vertex->coords.size(); ++joint)
        {
            coordinates.PushBack(vertex->coords[joint], allocator);
        }
        rapidjson::Value interval(rapidjson::kArrayType);
        interval.PushBack(vertex->safe_interval.first, allocator);
        interval.PushBack(vertex->safe_interval.second, allocator);
        item.AddMember("vertex_id", static_cast<uint64_t>(vertex->vertex_id), allocator);
        item.AddMember("tree_id", vertex->tree_id, allocator);
        item.AddMember("prediction_version", static_cast<uint64_t>(vertex->prediction_version), allocator);
        item.AddMember("arrival_frame", vertex->arrival_time, allocator);
        item.AddMember("departure_from_parent_frame", vertex->departure_from_parent_time, allocator);
        item.AddMember("safe_interval", interval, allocator);
        item.AddMember("robot_angles", coordinates, allocator);
        result.PushBack(item, allocator);
    }
    return result;
}

struct TruthEvaluation
{
    bool collision = false;
    double min_distance = std::numeric_limits<double>::infinity();
    std::size_t checked_frames = 0;
    int first_collision_frame = -1;
    int first_reactive_stop_frame = -1;
};

struct PathMetrics
{
    double joint_space_length = 0.0;
    double waiting_frames = 0.0;
    double arrival_frame = 0.0;
};

PathMetrics calculate_path_metrics(const std::vector<MDP::MSIRRT::Vertex *> &path)
{
    PathMetrics result;
    if (path.empty())
    {
        return result;
    }
    result.arrival_frame = path.back()->arrival_time;
    for (std::size_t index = 1; index < path.size(); ++index)
    {
        const auto *parent = path[index - 1];
        const auto *child = path[index];
        result.joint_space_length += (child->coords - parent->coords).norm();
        result.waiting_frames += std::max(0.0, child->departure_from_parent_time - parent->arrival_time);
    }
    return result;
}

struct PredictionCoverage
{
    std::size_t covered_samples = 0;
    std::size_t total_samples = 0;
    std::size_t simultaneously_covered_frames = 0;
    std::size_t total_frames = 0;
    double center_error_sum = 0.0;
    double maximum_center_error = 0.0;
    double tube_radius_sum = 0.0;
    double maximum_tube_radius = 0.0;
};

PredictionCoverage evaluate_prediction_coverage(
    const MDP::ConfigReader::SceneTask &prediction_scene,
    const MDP::ConfigReader::SceneTask &truth_scene,
    int first_frame,
    int last_frame)
{
    PredictionCoverage result;
    if (prediction_scene.obstacles.size() != truth_scene.obstacles.size())
    {
        throw std::runtime_error("prediction and truth obstacle counts differ");
    }
    first_frame = std::max(0, first_frame);
    last_frame = std::min(
        last_frame,
        static_cast<int>(std::min(prediction_scene.frame_count, truth_scene.frame_count)) - 1);
    if (last_frame < first_frame)
    {
        return result;
    }
    const std::size_t frame_count = static_cast<std::size_t>(last_frame - first_frame + 1);
    std::vector<bool> frame_covered(frame_count, true);
    std::vector<bool> frame_has_dynamic_sphere(frame_count, false);
    for (std::size_t obstacle_id = 0; obstacle_id < prediction_scene.obstacles.size(); ++obstacle_id)
    {
        const auto &prediction_obstacle = prediction_scene.obstacles[obstacle_id];
        const auto &truth_obstacle = truth_scene.obstacles[obstacle_id];
        if (prediction_obstacle->get_name() != truth_obstacle->get_name() ||
            prediction_obstacle->get_obstacle_type() != truth_obstacle->get_obstacle_type())
        {
            throw std::runtime_error("prediction and truth obstacle identities differ");
        }
        if (prediction_obstacle->get_is_static() ||
            prediction_obstacle->get_obstacle_type() != MDP::ObstacleJsonInfo::ObstacleType::SPHERE)
        {
            continue;
        }
        const auto *prediction_sphere = static_cast<const MDP::SphereObstacleJsonInfo *>(prediction_obstacle.get());
        const auto *truth_sphere = static_cast<const MDP::SphereObstacleJsonInfo *>(truth_obstacle.get());
        const auto &prediction_positions = prediction_sphere->get_coordinates();
        const auto &truth_positions = truth_sphere->get_coordinates();
        for (int frame = first_frame; frame <= last_frame; ++frame)
        {
            const auto &predicted = prediction_positions[frame];
            const auto &truth = truth_positions[frame];
            const double delta_x = static_cast<double>(predicted.x) - truth.x;
            const double delta_y = static_cast<double>(predicted.y) - truth.y;
            const double delta_z = static_cast<double>(predicted.z) - truth.z;
            const double center_error = std::sqrt(
                delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
            const double tube_radius = prediction_sphere->get_uncertainty_radius(frame);
            const double available_margin = std::max(
                0.0,
                static_cast<double>(prediction_sphere->get_radius()) + tube_radius - truth_sphere->get_radius());
            const auto relative_frame = static_cast<std::size_t>(frame - first_frame);
            frame_has_dynamic_sphere[relative_frame] = true;
            frame_covered[relative_frame] =
                frame_covered[relative_frame] && center_error <= available_margin + 1e-9;
            ++result.total_samples;
            result.covered_samples += center_error <= available_margin + 1e-9 ? 1 : 0;
            result.center_error_sum += center_error;
            result.maximum_center_error = std::max(result.maximum_center_error, center_error);
            result.tube_radius_sum += tube_radius;
            result.maximum_tube_radius = std::max(result.maximum_tube_radius, tube_radius);
        }
    }
    for (std::size_t frame = 0; frame < frame_count; ++frame)
    {
        if (!frame_has_dynamic_sphere[frame])
        {
            continue;
        }
        ++result.total_frames;
        result.simultaneously_covered_frames += frame_covered[frame] ? 1 : 0;
    }
    return result;
}

TruthEvaluation evaluate_against_truth(
    const std::vector<MDP::MSIRRT::Vertex *> &path,
    MDP::CollisionManager &truth_collision_manager,
    const MDP::ConfigReader::SceneTask &truth_scene,
    int evaluation_first_frame = 0,
    int evaluation_last_frame = std::numeric_limits<int>::max(),
    double reactive_stop_distance = -1.0)
{
    TruthEvaluation result;
    if (path.empty())
    {
        return result;
    }

    std::vector<bool> checked(truth_scene.frame_count, false);
    bool reactively_stopped = false;
    auto check_configuration = [&](const MDP::MSIRRT::Vertex::VertexCoordType &coords, int frame) {
        if (reactively_stopped || frame < evaluation_first_frame || frame > evaluation_last_frame ||
            frame < 0 || frame >= static_cast<int>(truth_scene.frame_count) || checked[frame])
        {
            return;
        }
        checked[frame] = true;
        ++result.checked_frames;
        const std::vector<double> angles(coords.data(), coords.data() + coords.size());
        if (truth_collision_manager.check_collision_frame_no_wrapper(angles, frame))
        {
            result.collision = true;
            if (result.first_collision_frame < 0)
            {
                result.first_collision_frame = frame;
            }
        }
        bool distance_collision = false;
        const auto distances = truth_collision_manager.get_distances(
            angles,
            static_cast<float>(frame) / static_cast<float>(truth_scene.fps),
            distance_collision);
        result.collision = result.collision || distance_collision;
        if (distance_collision && result.first_collision_frame < 0)
        {
            result.first_collision_frame = frame;
        }
        double frame_min_distance = std::numeric_limits<double>::infinity();
        for (const auto &distance : distances)
        {
            frame_min_distance = std::min(
                frame_min_distance,
                static_cast<double>(distance.min_distance));
        }
        result.min_distance = std::min(result.min_distance, frame_min_distance);
        if (reactive_stop_distance >= 0.0 && frame_min_distance <= reactive_stop_distance)
        {
            result.first_reactive_stop_frame = frame;
            reactively_stopped = true;
        }
    };

    check_configuration(path.front()->coords, static_cast<int>(std::floor(path.front()->arrival_time)));
    for (std::size_t index = 1; index < path.size(); ++index)
    {
        const auto *parent = path[index - 1];
        const auto *child = path[index];
        const int first_frame = std::max(
            evaluation_first_frame,
            std::max(0, static_cast<int>(std::floor(parent->arrival_time))));
        const int last_frame = std::min(
            evaluation_last_frame,
            std::min(
                static_cast<int>(truth_scene.frame_count) - 1,
                static_cast<int>(std::ceil(child->arrival_time))));
        for (int frame = first_frame; frame <= last_frame; ++frame)
        {
            if (frame <= child->departure_from_parent_time)
            {
                check_configuration(parent->coords, frame);
                continue;
            }
            const double duration = child->arrival_time - child->departure_from_parent_time;
            const double alpha = duration > 0.0
                ? std::clamp((frame - child->departure_from_parent_time) / duration, 0.0, 1.0)
                : 1.0;
            const auto coords = parent->coords + (child->coords - parent->coords) * alpha;
            check_configuration(coords, frame);
        }
    }
    return result;
}

void merge_truth_evaluation(TruthEvaluation &total, const TruthEvaluation &part)
{
    total.collision = total.collision || part.collision;
    total.checked_frames += part.checked_frames;
    total.min_distance = std::min(total.min_distance, part.min_distance);
    if (total.first_collision_frame < 0 && part.first_collision_frame >= 0)
    {
        total.first_collision_frame = part.first_collision_frame;
    }
    if (total.first_reactive_stop_frame < 0 && part.first_reactive_stop_frame >= 0)
    {
        total.first_reactive_stop_frame = part.first_reactive_stop_frame;
    }
}

struct BrakingEvaluation
{
    TruthEvaluation truth;
    std::vector<std::pair<int, std::vector<double>>> trajectory;
};

BrakingEvaluation evaluate_braking_trajectory(
    const std::vector<double> &current,
    const std::vector<double> &previous,
    int start_frame,
    int braking_frames,
    MDP::CollisionManager &truth_collision_manager,
    const MDP::ConfigReader::SceneTask &truth_scene)
{
    if (current.size() != previous.size() || braking_frames <= 0)
    {
        throw std::invalid_argument("braking trajectory inputs are invalid");
    }
    BrakingEvaluation result;
    const std::vector<double> velocity = [&]() {
        std::vector<double> value(current.size());
        for (std::size_t joint = 0; joint < current.size(); ++joint)
        {
            value[joint] = current[joint] - previous[joint];
        }
        return value;
    }();
    for (int step = 0; step <= braking_frames; ++step)
    {
        const int frame = start_frame + step;
        if (frame >= static_cast<int>(truth_scene.frame_count))
        {
            break;
        }
        const double displacement_scale = static_cast<double>(step) -
            static_cast<double>(step * step) / (2.0 * braking_frames);
        std::vector<double> configuration(current.size());
        for (std::size_t joint = 0; joint < current.size(); ++joint)
        {
            configuration[joint] = current[joint] + velocity[joint] * displacement_scale;
        }
        result.trajectory.emplace_back(frame, configuration);
        ++result.truth.checked_frames;
        if (truth_collision_manager.check_collision_frame_no_wrapper(configuration, frame))
        {
            result.truth.collision = true;
            if (result.truth.first_collision_frame < 0)
            {
                result.truth.first_collision_frame = frame;
            }
        }
        bool distance_collision = false;
        const auto distances = truth_collision_manager.get_distances(
            configuration,
            static_cast<float>(frame) / static_cast<float>(truth_scene.fps),
            distance_collision);
        result.truth.collision = result.truth.collision || distance_collision;
        if (distance_collision && result.truth.first_collision_frame < 0)
        {
            result.truth.first_collision_frame = frame;
        }
        for (const auto &distance : distances)
        {
            result.truth.min_distance = std::min(
                result.truth.min_distance,
                static_cast<double>(distance.min_distance));
        }
    }
    return result;
}

rapidjson::Value braking_trajectory_json(
    const BrakingEvaluation &braking,
    rapidjson::Document::AllocatorType &allocator)
{
    rapidjson::Value result(rapidjson::kArrayType);
    for (const auto &[frame, configuration] : braking.trajectory)
    {
        rapidjson::Value item(rapidjson::kObjectType);
        rapidjson::Value angles(rapidjson::kArrayType);
        for (const double angle : configuration)
        {
            angles.PushBack(angle, allocator);
        }
        item.AddMember("frame", frame, allocator);
        item.AddMember("robot_angles", angles, allocator);
        result.PushBack(item, allocator);
    }
    return result;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 9)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <manifest.json> <result.json> <seed> [max_updates]"
                  << " [repair|full_replan|rolling_repair|rolling_full_replan]"
                  << " [deadline_ms] [braking_frames] [reactive_stop_distance]\n";
        return 2;
    }

    try
    {
        const std::filesystem::path manifest_path = std::filesystem::absolute(argv[1]);
        const std::filesystem::path result_path = std::filesystem::absolute(argv[2]);
        const int seed = std::stoi(argv[3]);
        const std::size_t max_updates = argc >= 5 ? static_cast<std::size_t>(std::stoul(argv[4])) : 0;
        const std::string mode = argc >= 6 ? argv[5] : "repair";
        if (mode != "repair" && mode != "full_replan" &&
            mode != "rolling_repair" && mode != "rolling_full_replan")
        {
            throw std::runtime_error("unsupported planning mode");
        }
        const bool rolling = mode == "rolling_repair" || mode == "rolling_full_replan";
        const bool repair_mode = mode == "repair" || mode == "rolling_repair";
        const double deadline_ms = argc >= 7 ? std::stod(argv[6]) : 100.0;
        const int braking_frames = argc >= 8 ? std::stoi(argv[7]) : 2;
        const double reactive_stop_distance = argc >= 9 ? std::stod(argv[8]) : -1.0;
        if (!std::isfinite(deadline_ms) || deadline_ms <= 0.0 || braking_frames <= 0 ||
            !std::isfinite(reactive_stop_distance) || reactive_stop_distance < -1.0)
        {
            throw std::invalid_argument(
                "deadline_ms and braking_frames must be positive and reactive_stop_distance must be -1 or nonnegative");
        }
        const int progressive_initial_frames = positive_environment_integer(
            "MSIRRT_INITIAL_HORIZON_FRAMES", 0);
        const int progressive_step_frames = positive_environment_integer(
            "MSIRRT_HORIZON_STEP_FRAMES", 0);
        const int progressive_stage_budget_ms = positive_environment_integer(
            "MSIRRT_HORIZON_STAGE_BUDGET_MS", 20);
        const bool progressive_horizon_enabled =
            progressive_initial_frames > 0 && progressive_step_frames > 0;
        const char *policy_environment = std::getenv("MSIRRT_ADAPTIVE_REPAIR_POLICY");
        const std::string adaptive_repair_policy =
            policy_environment == nullptr ? "legacy" : policy_environment;
        if (adaptive_repair_policy != "legacy" && adaptive_repair_policy != "optimized")
        {
            throw std::invalid_argument(
                "MSIRRT_ADAPTIVE_REPAIR_POLICY must be legacy or optimized");
        }
        const bool feasibility_aware_horizon = adaptive_repair_policy == "legacy"
            ? false
            : environment_flag("MSIRRT_FEASIBILITY_AWARE_HORIZON", true);
        const int horizon_feasibility_slack_frames = nonnegative_environment_integer(
            "MSIRRT_HORIZON_FEASIBILITY_SLACK_FRAMES", 3);
        const double configured_stage_budget_reuse_fraction = bounded_environment_double(
            "MSIRRT_SKIPPED_STAGE_BUDGET_REUSE_FRACTION", 0.0, 0.0, 1.0);
        const double stage_budget_reuse_fraction = adaptive_repair_policy == "optimized"
            ? configured_stage_budget_reuse_fraction
            : 0.0;
        const double update_zero_final_guard_ms = bounded_environment_double(
            "MSIRRT_UPDATE_ZERO_FINAL_GUARD_MS", 15.0, 5.0, 15.0);
        const bool reuse_first_prediction_scene = environment_flag(
            "MSIRRT_REUSE_FIRST_PREDICTION_SCENE", false);
        const bool reuse_parsed_prediction_metadata = environment_flag(
            "MSIRRT_REUSE_PARSED_PREDICTION_METADATA", false);
        const bool cooperative_deadline_checks = environment_flag(
            "MSIRRT_COOPERATIVE_DEADLINE_CHECKS", false);
        const bool reachable_ellipsoid_sampling = environment_flag(
            "MSIRRT_REACHABLE_ELLIPSOID_SAMPLING", false);
        const bool fallback_enabled = environment_flag(
            "MSIRRT_FALLBACK_ENABLED", true);
        const char *tube_ablation_environment = std::getenv(
            "MSIRRT_TUBE_ABLATION_MODE");
        const std::string tube_ablation_mode = tube_ablation_environment == nullptr
            ? "adaptive"
            : tube_ablation_environment;
        if (tube_ablation_mode != "adaptive" &&
            tube_ablation_mode != "fixed_calibrated")
        {
            throw std::invalid_argument(
                "MSIRRT_TUBE_ABLATION_MODE must be adaptive or fixed_calibrated");
        }
        const bool fixed_tube_ablation_enabled =
            tube_ablation_mode == "fixed_calibrated";
        constexpr double fallback_reserve_fraction = 0.25;
        constexpr double fallback_reserve_min_ms = 10.0;
        constexpr double fallback_setup_estimate_ms = 15.0;
        constexpr double fallback_deadline_guard_ms = 5.0;
        constexpr double planning_call_guard_ms = 15.0;
        const rapidjson::Document manifest = read_json(manifest_path);
        if (!manifest.HasMember("prediction_files") || !manifest["prediction_files"].IsArray() ||
            manifest["prediction_files"].Empty() || !manifest.HasMember("truth_scene"))
        {
            throw std::runtime_error("manifest prediction_files must be a non-empty array");
        }
        MDP::MSIRRT::FixedTubeAblationConfig fixed_tube_config;
        if (fixed_tube_ablation_enabled)
        {
            fixed_tube_config.fixed_radius = required_manifest_config_double(
                manifest, "fixed_radius");
            fixed_tube_config.maximum_allowed_radius =
                required_manifest_config_double(manifest, "max_tube_radius");
            fixed_tube_config.maximum_prediction_horizon_frames = static_cast<int>(
                required_manifest_config_double(
                    manifest, "max_prediction_horizon_frames"));
        }

        std::vector<std::filesystem::path> prediction_paths;
        for (const auto &entry : manifest["prediction_files"].GetArray())
        {
            prediction_paths.push_back(manifest_path.parent_path() / entry.GetString());
        }
        if (max_updates > 0 && prediction_paths.size() > max_updates)
        {
            prediction_paths.resize(max_updates);
        }

        const auto initial_planner_setup_start = std::chrono::steady_clock::now();
        const auto initial_prediction_start = std::chrono::steady_clock::now();
        std::string initial_prediction_status = "unspecified";
        if (reuse_first_prediction_scene && !reuse_parsed_prediction_metadata)
        {
            initial_prediction_status = prediction_status_from_document(
                read_json(prediction_paths.front()));
        }
        MDP::ConfigReader initial_reader(prediction_paths.front().string());
        const auto initial_maximum_scene = initial_reader.get_scene_task();
        if (reuse_first_prediction_scene && reuse_parsed_prediction_metadata)
        {
            initial_prediction_status = initial_maximum_scene.prediction_status;
        }
        const double initial_prediction_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - initial_prediction_start).count();
        const auto initial_tube_update_start = std::chrono::steady_clock::now();
        auto initial_scene = initial_maximum_scene;
        MDP::MSIRRT::FixedTubeAblationReport initial_fixed_tube_report;
        if (fixed_tube_ablation_enabled)
        {
            initial_fixed_tube_report = MDP::MSIRRT::apply_fixed_calibrated_tube(
                initial_scene, fixed_tube_config);
        }
        const int initial_maximum_reliable_horizon_frame = static_cast<int>(
            initial_scene.reliable_until_frame);
        MDP::MSIRRT::InitialHorizonSelection first_horizon_selection;
        if (progressive_horizon_enabled)
        {
            const int minimum_initial_horizon = MDP::MSIRRT::minimum_kinematic_arrival_frame(
                initial_scene.start_configuration,
                initial_scene.end_configuration,
                static_cast<int>(initial_scene.prediction_issue_frame),
                static_cast<int>(initial_scene.fps),
                MDP::MSIRRT::DEFAULT_MAXIMUM_JOINT_SPACE_SPEED);
            first_horizon_selection = MDP::MSIRRT::select_initial_planning_horizon(
                static_cast<int>(initial_scene.prediction_issue_frame),
                progressive_initial_frames,
                minimum_initial_horizon,
                initial_maximum_reliable_horizon_frame,
                horizon_feasibility_slack_frames,
                progressive_step_frames,
                feasibility_aware_horizon);
            initial_scene.reliable_until_frame = static_cast<unsigned int>(
                first_horizon_selection.selected_frame);
        }
        const double initial_tube_update_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - initial_tube_update_start).count();
        const auto initial_safe_interval_update_start = std::chrono::steady_clock::now();
        auto planner = std::make_unique<MDP::MSIRRT::PlannerConnect>(initial_scene, seed);
        if (planner->get_cooperative_deadline_checks_enabled() !=
            cooperative_deadline_checks)
        {
            throw std::runtime_error("cooperative deadline configuration mismatch");
        }
        if (planner->get_reachable_ellipsoid_sampling_enabled() !=
            reachable_ellipsoid_sampling)
        {
            throw std::runtime_error("reachable ellipsoid sampling configuration mismatch");
        }
        const double initial_safe_interval_update_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - initial_safe_interval_update_start).count();
        const double initial_planner_setup_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - initial_planner_setup_start).count();
        const double attributed_initial_setup_seconds =
            initial_prediction_seconds + initial_tube_update_seconds +
            initial_safe_interval_update_seconds;
        if (initial_planner_setup_seconds - attributed_initial_setup_seconds > 0.0001)
        {
            throw std::runtime_error("initial planner setup timing does not reconcile");
        }
        auto replace_planner = [&](const MDP::ConfigReader::SceneTask &replacement_scene,
                                   int replacement_seed,
                                   double &setup_seconds) {
            const auto setup_start = std::chrono::steady_clock::now();
            try
            {
                auto replacement = std::make_unique<MDP::MSIRRT::PlannerConnect>(
                    replacement_scene, replacement_seed);
                planner = std::move(replacement);
                setup_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - setup_start).count();
                return true;
            }
            catch (const std::runtime_error &error)
            {
                setup_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - setup_start).count();
                if (std::string(error.what()) !=
                    "start configuration is not safe at prediction issue frame")
                {
                    throw;
                }
                return false;
            }
        };
        const std::filesystem::path truth_path = manifest_path.parent_path() / manifest["truth_scene"].GetString();
        MDP::ConfigReader truth_reader(truth_path.string());
        MDP::CollisionManager truth_collision_manager(truth_reader.get_scene_task());

        rapidjson::Document output;
        output.SetObject();
        auto &allocator = output.GetAllocator();
        output.AddMember("schema_version", 1, allocator);
        output.AddMember("seed", seed, allocator);
        output.AddMember("mode", rapidjson::Value(mode.c_str(), allocator), allocator);
        output.AddMember("rolling_execution", rolling, allocator);
        output.AddMember("deadline_ms", deadline_ms, allocator);
        output.AddMember("braking_frames", braking_frames, allocator);
        output.AddMember("reactive_stop_distance", reactive_stop_distance, allocator);
        output.AddMember("progressive_horizon_enabled", progressive_horizon_enabled, allocator);
        output.AddMember(
            "adaptive_repair_policy",
            rapidjson::Value(adaptive_repair_policy.c_str(), allocator),
            allocator);
        output.AddMember("feasibility_aware_horizon", feasibility_aware_horizon, allocator);
        output.AddMember(
            "horizon_feasibility_slack_frames",
            horizon_feasibility_slack_frames,
            allocator);
        output.AddMember("initial_horizon_frames", progressive_initial_frames, allocator);
        output.AddMember("horizon_step_frames", progressive_step_frames, allocator);
        output.AddMember("horizon_stage_budget_ms", progressive_stage_budget_ms, allocator);
        output.AddMember(
            "stage_budget_reuse_fraction",
            stage_budget_reuse_fraction,
            allocator);
        output.AddMember(
            "reuse_first_prediction_scene",
            reuse_first_prediction_scene,
            allocator);
        output.AddMember(
            "reuse_parsed_prediction_metadata",
            reuse_parsed_prediction_metadata,
            allocator);
        output.AddMember(
            "cooperative_deadline_checks",
            cooperative_deadline_checks,
            allocator);
        output.AddMember(
            "reachable_ellipsoid_sampling",
            reachable_ellipsoid_sampling,
            allocator);
        output.AddMember(
            "horizon_bounded_safe_interval_queries",
            planner->get_horizon_bounded_safe_interval_queries_enabled(),
            allocator);
        output.AddMember("fallback_enabled", fallback_enabled, allocator);
        output.AddMember(
            "tube_ablation_mode",
            rapidjson::Value(tube_ablation_mode.c_str(), allocator),
            allocator);
        if (fixed_tube_ablation_enabled)
        {
            output.AddMember("fixed_tube_radius", fixed_tube_config.fixed_radius, allocator);
            output.AddMember(
                "fixed_tube_maximum_allowed_radius",
                fixed_tube_config.maximum_allowed_radius,
                allocator);
            output.AddMember(
                "fixed_tube_maximum_prediction_horizon_frames",
                fixed_tube_config.maximum_prediction_horizon_frames,
                allocator);
            output.AddMember(
                "fixed_tube_transformed_dynamic_spheres",
                static_cast<uint64_t>(initial_fixed_tube_report.transformed_dynamic_spheres),
                allocator);
        }
        output.AddMember("fallback_reserve_fraction", fallback_reserve_fraction, allocator);
        output.AddMember("fallback_reserve_min_ms", fallback_reserve_min_ms, allocator);
        output.AddMember("fallback_setup_estimate_ms", fallback_setup_estimate_ms, allocator);
        output.AddMember("fallback_deadline_guard_ms", fallback_deadline_guard_ms, allocator);
        output.AddMember("planning_call_guard_ms", planning_call_guard_ms, allocator);
        output.AddMember(
            "update_zero_final_guard_ms", update_zero_final_guard_ms, allocator);
        output.AddMember(
            "latency_accounting",
            rapidjson::Value("planning_decision_wall_clock", allocator),
            allocator);
        output.AddMember(
            "manifest",
            rapidjson::Value(manifest_path.string().c_str(), allocator),
            allocator);
        rapidjson::Value updates(rapidjson::kArrayType);
        TruthEvaluation executed_truth;
        bool task_completed = false;
        bool emergency_stop = false;
        bool braking_collision = false;
        bool ever_accepted_plan = false;
        bool any_deadline_missed = false;
        bool any_prediction_unavailable = false;
        bool any_reliable_horizon_too_short = false;
        bool any_repair_invariant_failure = false;
        bool any_stale_prefix_after_invalidation = false;
        bool any_returned_path_current_scene_validation_failure = false;
        std::size_t terminal_update_index = 0;
        double completion_frame = std::numeric_limits<double>::quiet_NaN();
        std::string termination_reason = rolling ? "prediction_sequence_exhausted" : "not_applicable";

        for (std::size_t update_index = 0; update_index < prediction_paths.size(); ++update_index)
        {
            terminal_update_index = update_index;
            const auto update_wall_start = std::chrono::steady_clock::now();
            const double carried_initial_setup_seconds = update_index == 0
                ? initial_planner_setup_seconds
                : 0.0;
            MDP::MSIRRT::OnlineStageTiming stage_timing;
            if (update_index == 0)
            {
                stage_timing.prediction_ms = 1000.0 * initial_prediction_seconds;
                stage_timing.tube_update_ms = 1000.0 * initial_tube_update_seconds;
                stage_timing.safe_interval_update_ms =
                    1000.0 * initial_safe_interval_update_seconds;
            }
            double excluded_truth_evaluation_seconds = 0.0;
            const auto planning_elapsed_seconds = [&]() {
                return std::max(
                    0.0,
                    carried_initial_setup_seconds + std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - update_wall_start).count() -
                        excluded_truth_evaluation_seconds);
            };
            const auto measure_excluded_truth_evaluation = [&](auto &&evaluation) {
                const auto evaluation_start = std::chrono::steady_clock::now();
                auto value = evaluation();
                excluded_truth_evaluation_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - evaluation_start).count();
                return value;
            };
            const auto update_prediction_start = std::chrono::steady_clock::now();
            const bool used_cached_prediction_scene =
                reuse_first_prediction_scene && update_index == 0;
            std::string prediction_status = initial_prediction_status;
            if (!used_cached_prediction_scene && !reuse_parsed_prediction_metadata)
            {
                prediction_status = prediction_status_from_document(
                    read_json(prediction_paths[update_index]));
            }
            const auto maximum_scene = [&]() {
                if (used_cached_prediction_scene)
                {
                    return initial_maximum_scene;
                }
                MDP::ConfigReader reader(prediction_paths[update_index].string());
                return reader.get_scene_task();
            }();
            if (!used_cached_prediction_scene && reuse_parsed_prediction_metadata)
            {
                prediction_status = maximum_scene.prediction_status;
            }
            stage_timing.prediction_ms += 1000.0 * std::chrono::duration<double>(
                std::chrono::steady_clock::now() - update_prediction_start).count();
            const auto update_tube_start = std::chrono::steady_clock::now();
            auto scene = maximum_scene;
            MDP::MSIRRT::FixedTubeAblationReport fixed_tube_report;
            if (fixed_tube_ablation_enabled)
            {
                fixed_tube_report = MDP::MSIRRT::apply_fixed_calibrated_tube(
                    scene, fixed_tube_config);
            }
            const int maximum_reliable_horizon_frame = static_cast<int>(
                scene.reliable_until_frame);
            if (progressive_horizon_enabled)
            {
                scene.reliable_until_frame = static_cast<unsigned int>(std::min(
                    maximum_reliable_horizon_frame,
                    static_cast<int>(scene.prediction_issue_frame) + progressive_initial_frames));
            }
            stage_timing.tube_update_ms += 1000.0 * std::chrono::duration<double>(
                std::chrono::steady_clock::now() - update_tube_start).count();
            const int raw_initial_planning_horizon_frame = static_cast<int>(
                scene.reliable_until_frame);
            int initial_planning_horizon_frame = raw_initial_planning_horizon_frame;
            int skipped_infeasible_horizon_stages = 0;
            std::vector<int> attempted_horizon_frames{initial_planning_horizon_frame};
            std::vector<double> horizon_stage_solve_ms;
            int horizon_expansions = 0;
            MDP::MSIRRT::BidirectionalRepairReport repair;
            repair.start_tree.invariants_hold = true;
            repair.goal_tree.invariants_hold = true;
            repair.invariants_hold = true;
            MDP::MSIRRT::ExecutionAdvanceReport execution_advance;
            TruthEvaluation executed_prefix_truth;
            BrakingEvaluation braking;
            std::vector<double> current_configuration = scene.start_configuration;
            std::vector<double> previous_configuration = current_configuration;
            bool can_plan = true;
            bool accepted_plan = false;
            bool fallback_full_replan_used = false;
            bool maximum_horizon_budget_continuation_used = false;
            bool reliable_horizon_too_short = false;
            bool horizon_feasible = true;
            double fallback_setup_seconds = 0.0;
            double fallback_solve_seconds = 0.0;
            double fallback_seconds = 0.0;
            double fallback_preflight_solve_budget_ms = 0.0;
            double progressive_setup_seconds = 0.0;
            double saved_infeasible_stage_budget_ms = 0.0;
            double initial_stage_budget_limit_ms = 0.0;
            double progressive_guard_remaining_ms = 0.0;
            double final_stage_effective_guard_ms = planning_call_guard_ms;
            double final_stage_solve_budget_ms = 0.0;
            bool final_stage_continuation_used = false;
            MDP::MSIRRT::PlanningBudget planning_budget;
            int minimum_required_horizon_frame = static_cast<int>(
                scene.prediction_issue_frame);
            int braking_start_frame = static_cast<int>(scene.prediction_issue_frame);
            int reactive_stop_frame = -1;
            std::string control_state = "planning";
            std::string budget_exhaustion_stage = "none";
            MDP::MSIRRT::ExecutionWindow execution_window;
            execution_window.requested_frame = static_cast<int>(scene.prediction_issue_frame);
            execution_window.reliable_until_frame =
                static_cast<int>(scene.prediction_issue_frame);
            execution_window.path_until_frame = static_cast<int>(scene.prediction_issue_frame);
            execution_window.validated_frame = static_cast<int>(scene.prediction_issue_frame);
            int actual_execution_frame = static_cast<int>(scene.prediction_issue_frame);
            std::size_t evaluated_before = planner->get_evaluated_samples();
            std::size_t rejected_before = planner->get_conditionally_rejected_samples();
            std::size_t rejected_kinematic_before = planner->get_conditionally_rejected_kinematic_samples();
            std::size_t rejected_temporal_before = planner->get_conditionally_rejected_temporal_samples();
            std::size_t bypassed_before = planner->get_conditionally_bypassed_samples();
            std::size_t proposed_before = planner->get_proposed_samples();
            std::size_t feedback_triggers_before =
                planner->get_conditional_rejection_feedback_triggers();
            std::size_t feedback_boosted_before =
                planner->get_conditional_feedback_boosted_samples();
            std::size_t cooperative_deadline_check_triggers_before =
                planner->get_cooperative_deadline_check_triggers();
            std::size_t reachable_ellipsoid_samples_before =
                planner->get_reachable_ellipsoid_samples();
            int planner_seed = seed;
            const auto stop_for_unsafe_current_configuration = [&]() {
                can_plan = false;
                emergency_stop = true;
                control_state = "safe_stop_current_configuration_unsafe";
                termination_reason = "current_configuration_unsafe_in_prediction";
            };

            if (rolling && prediction_status == "unavailable")
            {
                braking_start_frame = planner->get_planning_start_frame();
                current_configuration = planner->configuration_at_frame(braking_start_frame);
                previous_configuration = current_configuration;
                can_plan = false;
                emergency_stop = true;
                control_state = "safe_stop_prediction_unavailable";
                termination_reason = "prediction_unavailable";
            }
            else if (update_index > 0)
            {
                if (rolling)
                {
                    const auto previous_path = planner->get_final_path();
                    const int path_until_frame = previous_path.empty()
                        ? planner->get_planning_start_frame()
                        : static_cast<int>(std::ceil(previous_path.back()->arrival_time));
                    execution_window = MDP::MSIRRT::select_execution_window(
                        planner->get_planning_start_frame(),
                        static_cast<int>(scene.prediction_issue_frame),
                        planner->get_planning_horizon_frame(),
                        path_until_frame);
                    if (previous_path.empty())
                    {
                        can_plan = false;
                        emergency_stop = true;
                        control_state = "safe_stop_missing_previous_path";
                        termination_reason = "missing_previous_path";
                    }
                    else if (execution_window.prefix_frames == 0)
                    {
                        can_plan = false;
                        emergency_stop = true;
                        control_state = "safe_stop_no_executable_prefix";
                        termination_reason = "no_executable_prefix";
                    }
                    else if (execution_window.validated_frame >= previous_path.back()->arrival_time)
                    {
                        actual_execution_frame = path_until_frame;
                        const auto final_execution = measure_excluded_truth_evaluation([&]() {
                            return evaluate_against_truth(
                                previous_path,
                                truth_collision_manager,
                                truth_reader.get_scene_task(),
                                planner->get_planning_start_frame(),
                                static_cast<int>(std::ceil(previous_path.back()->arrival_time)),
                                reactive_stop_distance);
                        });
                        merge_truth_evaluation(executed_truth, final_execution);
                        if (final_execution.first_reactive_stop_frame >= 0)
                        {
                            const int stop_frame = final_execution.first_reactive_stop_frame;
                            actual_execution_frame = stop_frame;
                            const auto stop_configuration = planner->configuration_at_frame(stop_frame);
                            const auto previous_stop_configuration = planner->configuration_at_frame(
                                std::max(planner->get_planning_start_frame(), stop_frame - 1));
                            braking = measure_excluded_truth_evaluation([&]() {
                                return evaluate_braking_trajectory(
                                    stop_configuration,
                                    previous_stop_configuration,
                                    stop_frame,
                                    braking_frames,
                                    truth_collision_manager,
                                    truth_reader.get_scene_task());
                            });
                            emergency_stop = true;
                            braking_collision = braking.truth.collision;
                            termination_reason = "reactive_clearance_threshold";
                        }
                        else
                        {
                            task_completed = !final_execution.collision;
                            completion_frame = previous_path.back()->arrival_time;
                            termination_reason = final_execution.collision
                                ? "truth_collision_before_goal"
                                : "goal_reached";
                        }
                        break;
                    }
                    else
                    {
                        const int execution_frame = execution_window.validated_frame;
                        actual_execution_frame = execution_frame;
                        braking_start_frame = execution_frame;
                        executed_prefix_truth = measure_excluded_truth_evaluation([&]() {
                            return evaluate_against_truth(
                                previous_path,
                                truth_collision_manager,
                                truth_reader.get_scene_task(),
                                planner->get_planning_start_frame(),
                                execution_frame,
                                reactive_stop_distance);
                        });
                        merge_truth_evaluation(executed_truth, executed_prefix_truth);
                        if (executed_prefix_truth.first_reactive_stop_frame >= 0)
                        {
                            reactive_stop_frame = executed_prefix_truth.first_reactive_stop_frame;
                            actual_execution_frame = reactive_stop_frame;
                            braking_start_frame = reactive_stop_frame;
                            current_configuration = planner->configuration_at_frame(reactive_stop_frame);
                            previous_configuration = planner->configuration_at_frame(
                                std::max(planner->get_planning_start_frame(), reactive_stop_frame - 1));
                            can_plan = false;
                            emergency_stop = true;
                            control_state = "safe_stop_reactive_clearance";
                            termination_reason = "reactive_clearance_threshold";
                        }
                        else
                        {
                            current_configuration = planner->configuration_at_frame(execution_frame);
                            previous_configuration = planner->configuration_at_frame(
                                std::max(planner->get_planning_start_frame(), execution_frame - 1));
                            if (executed_prefix_truth.collision)
                            {
                                can_plan = false;
                                emergency_stop = true;
                                control_state = "safe_stop_truth_execution_collision";
                                termination_reason = "truth_execution_collision";
                            }
                            else if (execution_window.limited_by_reliability)
                            {
                                can_plan = false;
                                emergency_stop = true;
                                control_state = "safe_stop_prediction_gap";
                                termination_reason = "prediction_gap_before_next_update";
                            }
                            else if (repair_mode)
                            {
                                execution_advance = planner->advance_start_to_frame(execution_frame);
                                scene.start_configuration = execution_advance.configuration;
                                repair = planner->update_prediction(scene);
                            }
                            else
                            {
                                scene.start_configuration = current_configuration;
                                planner_seed = seed + static_cast<int>(update_index);
                                double setup_seconds = 0.0;
                                if (!replace_planner(scene, planner_seed, setup_seconds))
                                {
                                    stop_for_unsafe_current_configuration();
                                }
                                repair.collision_structure_update_seconds = setup_seconds;
                                repair.safe_interval_update_seconds = setup_seconds;
                                repair.collision_frames_rebuilt = scene.frame_count;
                                repair.collision_obstacles_reindexed = scene.obstacles.size();
                                repair.collision_full_rebuild_fallback = true;
                                if (can_plan)
                                {
                                    evaluated_before = 0;
                                    rejected_before = 0;
                                    rejected_kinematic_before = 0;
                                    rejected_temporal_before = 0;
                                    bypassed_before = 0;
                                    proposed_before = 0;
                                    feedback_triggers_before = 0;
                                    feedback_boosted_before = 0;
                                    cooperative_deadline_check_triggers_before = 0;
                                    reachable_ellipsoid_samples_before = 0;
                                }
                            }
                        }
                    }
                }
                else if (repair_mode)
                {
                    repair = planner->update_prediction(scene);
                }
                else
                {
                    planner_seed = seed + static_cast<int>(update_index);
                    double setup_seconds = 0.0;
                    if (!replace_planner(scene, planner_seed, setup_seconds))
                    {
                        throw std::runtime_error(
                            "start configuration is not safe at prediction issue frame");
                    }
                    repair.collision_structure_update_seconds = setup_seconds;
                    repair.safe_interval_update_seconds = setup_seconds;
                    repair.collision_frames_rebuilt = scene.frame_count;
                    repair.collision_obstacles_reindexed = scene.obstacles.size();
                    repair.collision_full_rebuild_fallback = true;
                    repair.start_tree.invariants_hold = true;
                    repair.goal_tree.invariants_hold = true;
                    repair.invariants_hold = true;
                    evaluated_before = 0;
                    rejected_before = 0;
                    rejected_kinematic_before = 0;
                    rejected_temporal_before = 0;
                    bypassed_before = 0;
                    proposed_before = 0;
                    feedback_triggers_before = 0;
                    feedback_boosted_before = 0;
                    cooperative_deadline_check_triggers_before = 0;
                    reachable_ellipsoid_samples_before = 0;
                }
            }
            minimum_required_horizon_frame = MDP::MSIRRT::minimum_kinematic_arrival_frame(
                scene.start_configuration,
                scene.end_configuration,
                planner->get_planning_start_frame(),
                static_cast<int>(scene.fps),
                planner->get_maximum_joint_space_speed());
            horizon_feasible = minimum_required_horizon_frame <= maximum_reliable_horizon_frame;
            const auto horizon_selection = MDP::MSIRRT::select_initial_planning_horizon(
                static_cast<int>(scene.prediction_issue_frame),
                progressive_horizon_enabled ? progressive_initial_frames :
                    maximum_reliable_horizon_frame - static_cast<int>(scene.prediction_issue_frame),
                minimum_required_horizon_frame,
                maximum_reliable_horizon_frame,
                horizon_feasibility_slack_frames,
                progressive_horizon_enabled ? progressive_step_frames : 1,
                progressive_horizon_enabled && feasibility_aware_horizon);
            initial_planning_horizon_frame = horizon_selection.selected_frame;
            skipped_infeasible_horizon_stages =
                horizon_selection.skipped_infeasible_stages;
            if (can_plan && horizon_selection.selected_frame >
                    static_cast<int>(scene.reliable_until_frame))
            {
                scene.reliable_until_frame = static_cast<unsigned int>(
                    horizon_selection.selected_frame);
                attempted_horizon_frames.front() = horizon_selection.selected_frame;
                if (update_index > 0 && repair_mode)
                {
                    const auto feasibility_expansion = planner->update_prediction(scene);
                    accumulate_bidirectional_repair(repair, feasibility_expansion);
                }
                else if (update_index > 0)
                {
                    double setup_seconds = 0.0;
                    if (!replace_planner(scene, planner_seed, setup_seconds))
                    {
                        stop_for_unsafe_current_configuration();
                    }
                    progressive_setup_seconds += setup_seconds;
                    repair.safe_interval_update_seconds += setup_seconds;
                }
            }
            if (rolling && can_plan &&
                !horizon_feasible)
            {
                can_plan = false;
                emergency_stop = true;
                reliable_horizon_too_short = true;
                control_state = "safe_stop_reliable_horizon_too_short";
                termination_reason = "reliable_horizon_too_short";
            }

            const double update_seconds_before_solve = planning_elapsed_seconds();
            const bool fallback_applicable = fallback_enabled && rolling && repair_mode && update_index > 0 &&
                prediction_status != "unavailable";
            planning_budget = MDP::MSIRRT::allocate_planning_budget(
                deadline_ms,
                1000.0 * update_seconds_before_solve,
                fallback_applicable ? fallback_reserve_fraction : 0.0,
                fallback_applicable ? fallback_reserve_min_ms : 0.0);
            double available_solve_seconds =
                planning_budget.available_after_update_ms / 1000.0;
            const double progressive_budget_seconds =
                planning_budget.progressive_ms / 1000.0;
            if (progressive_horizon_enabled)
            {
                saved_infeasible_stage_budget_ms =
                    static_cast<double>(progressive_stage_budget_ms) *
                    static_cast<double>(skipped_infeasible_horizon_stages) *
                    stage_budget_reuse_fraction;
                initial_stage_budget_limit_ms = MDP::MSIRRT::initial_stage_budget_ms(
                    progressive_stage_budget_ms,
                    skipped_infeasible_horizon_stages,
                    stage_budget_reuse_fraction,
                    planning_budget.progressive_ms,
                    planning_call_guard_ms);
                progressive_guard_remaining_ms = std::max(
                    0.0,
                    planning_budget.progressive_ms - planning_call_guard_ms -
                        initial_stage_budget_limit_ms);
            }
            else
            {
                initial_stage_budget_limit_ms = planning_budget.progressive_ms;
            }
            const auto progressive_deadline = std::chrono::steady_clock::now() +
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(progressive_budget_seconds));
            if (rolling && can_plan && available_solve_seconds <= 0.0)
            {
                can_plan = false;
                emergency_stop = true;
                control_state = "safe_stop_deadline_exhausted_before_solve";
                termination_reason = "deadline_exhausted_before_solve";
                budget_exhaustion_stage = "before_solve";
            }
            double initial_budget = progressive_budget_seconds;
            const bool preserved_solution_fast_path =
                can_plan && planner->has_valid_solution();
            if (rolling && can_plan && !preserved_solution_fast_path)
            {
                initial_budget = progressive_horizon_enabled
                    ? initial_stage_budget_limit_ms / 1000.0
                    : progressive_budget_seconds;
                if (initial_budget > 0.0001)
                {
                    planner->set_max_planning_time(initial_budget);
                }
                else
                {
                    budget_exhaustion_stage = "initial_guard";
                    if (!fallback_applicable)
                    {
                        can_plan = false;
                        emergency_stop = true;
                        control_state = "safe_stop_deadline_exhausted_before_solve";
                        termination_reason = "deadline_exhausted_before_solve";
                    }
                }
            }
            bool solved = preserved_solution_fast_path;
            double solve_seconds = 0.0;
            if (can_plan && !solved && (!rolling || initial_budget > 0.0001))
            {
                planner->reset_conditional_exploration_feedback_stage();
                const auto solve_start = std::chrono::steady_clock::now();
                solved = planner->solve();
                solve_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - solve_start).count();
                horizon_stage_solve_ms.push_back(1000.0 * solve_seconds);
            }

            while (progressive_horizon_enabled && rolling && can_plan && !solved &&
                   static_cast<int>(scene.reliable_until_frame) < maximum_reliable_horizon_frame)
            {
                double remaining_seconds = std::chrono::duration<double>(
                    progressive_deadline - std::chrono::steady_clock::now()).count();
                if (remaining_seconds <= planning_call_guard_ms / 1000.0)
                {
                    budget_exhaustion_stage = "progressive_guard";
                    break;
                }
                const int next_horizon = std::min(
                    maximum_reliable_horizon_frame,
                    static_cast<int>(scene.reliable_until_frame) + progressive_step_frames);
                scene.reliable_until_frame = static_cast<unsigned int>(next_horizon);
                attempted_horizon_frames.push_back(next_horizon);
                ++horizon_expansions;

                if (repair_mode)
                {
                    const auto expansion_repair = planner->update_prediction(scene);
                    accumulate_bidirectional_repair(repair, expansion_repair);
                }
                else
                {
                    planner_seed = seed + static_cast<int>(update_index + horizon_expansions);
                    double setup_seconds = 0.0;
                    if (!replace_planner(scene, planner_seed, setup_seconds))
                    {
                        stop_for_unsafe_current_configuration();
                    }
                    progressive_setup_seconds += setup_seconds;
                    repair.collision_frames_rebuilt += scene.frame_count;
                    repair.collision_obstacles_reindexed += scene.obstacles.size();
                    repair.collision_full_rebuild_fallback = true;
                    if (can_plan)
                    {
                        evaluated_before = 0;
                        rejected_before = 0;
                        rejected_kinematic_before = 0;
                        rejected_temporal_before = 0;
                        bypassed_before = 0;
                        proposed_before = 0;
                        feedback_triggers_before = 0;
                        feedback_boosted_before = 0;
                        cooperative_deadline_check_triggers_before = 0;
                        reachable_ellipsoid_samples_before = 0;
                    }
                }

                if (!can_plan)
                {
                    break;
                }
                remaining_seconds = std::chrono::duration<double>(
                    progressive_deadline - std::chrono::steady_clock::now()).count();
                if (remaining_seconds <= planning_call_guard_ms / 1000.0)
                {
                    budget_exhaustion_stage = "progressive_guard";
                    break;
                }
                planner->set_max_planning_time(std::min(
                    remaining_seconds - planning_call_guard_ms / 1000.0,
                    static_cast<double>(progressive_stage_budget_ms) / 1000.0));
                planner->reset_conditional_exploration_feedback_stage();
                const auto stage_start = std::chrono::steady_clock::now();
                solved = planner->solve();
                const double stage_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - stage_start).count();
                solve_seconds += stage_seconds;
                horizon_stage_solve_ms.push_back(1000.0 * stage_seconds);
            }

            if (progressive_horizon_enabled && rolling && can_plan && !solved &&
                static_cast<int>(scene.reliable_until_frame) >= maximum_reliable_horizon_frame)
            {
                const double remaining_seconds = std::chrono::duration<double>(
                    progressive_deadline - std::chrono::steady_clock::now()).count();
                const auto final_stage_budget =
                    MDP::MSIRRT::select_search_continuation_budget(
                        std::max(0.0, 1000.0 * remaining_seconds),
                        planning_call_guard_ms,
                        update_zero_final_guard_ms,
                        update_index == 0,
                        static_cast<int>(scene.reliable_until_frame) >=
                            maximum_reliable_horizon_frame,
                        fallback_applicable);
                final_stage_effective_guard_ms =
                    final_stage_budget.effective_guard_ms;
                final_stage_solve_budget_ms = final_stage_budget.solve_budget_ms;
                if (final_stage_budget.allowed)
                {
                    maximum_horizon_budget_continuation_used = true;
                    final_stage_continuation_used = true;
                    planner->set_max_planning_time(
                        final_stage_budget.solve_budget_ms / 1000.0);
                    planner->reset_conditional_exploration_feedback_stage();
                    const auto continuation_start = std::chrono::steady_clock::now();
                    solved = planner->solve();
                    const double continuation_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - continuation_start).count();
                    solve_seconds += continuation_seconds;
                    horizon_stage_solve_ms.push_back(1000.0 * continuation_seconds);
                }
                else if (budget_exhaustion_stage == "none")
                {
                    budget_exhaustion_stage = "progressive_guard";
                }
            }

            if (fallback_enabled && rolling && repair_mode &&
                update_index > 0 && can_plan && !solved &&
                prediction_status != "unavailable")
            {
                const auto fallback_start = std::chrono::steady_clock::now();
                const double remaining_seconds = deadline_ms / 1000.0 -
                    planning_elapsed_seconds();
                fallback_preflight_solve_budget_ms =
                    MDP::MSIRRT::fallback_solve_budget_ms(
                        std::max(0.0, 1000.0 * remaining_seconds),
                        fallback_setup_estimate_ms,
                        fallback_deadline_guard_ms);
                if (fallback_preflight_solve_budget_ms > 0.0001)
                {
                    fallback_full_replan_used = true;
                    planner_seed = seed + static_cast<int>(update_index);
                    if (!replace_planner(scene, planner_seed, fallback_setup_seconds))
                    {
                        stop_for_unsafe_current_configuration();
                    }
                    else
                    {
                        evaluated_before = 0;
                        rejected_before = 0;
                        rejected_kinematic_before = 0;
                        rejected_temporal_before = 0;
                        bypassed_before = 0;
                        proposed_before = 0;
                        feedback_triggers_before = 0;
                        feedback_boosted_before = 0;
                        cooperative_deadline_check_triggers_before = 0;
                        reachable_ellipsoid_samples_before = 0;
                    }
                    repair.collision_frames_rebuilt += scene.frame_count;
                    repair.collision_obstacles_reindexed += scene.obstacles.size();
                    repair.collision_full_rebuild_fallback = true;
                    const double fallback_budget = deadline_ms / 1000.0 -
                        planning_elapsed_seconds() - 0.005;
                    if (can_plan && fallback_budget > 0.0001)
                    {
                        planner->set_max_planning_time(fallback_budget);
                        planner->reset_conditional_exploration_feedback_stage();
                        const auto fallback_solve_start = std::chrono::steady_clock::now();
                        solved = planner->solve();
                        fallback_solve_seconds = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - fallback_solve_start).count();
                        solve_seconds += fallback_solve_seconds;
                    }
                    else if (can_plan)
                    {
                        budget_exhaustion_stage = "fallback_setup";
                    }
                }
                else
                {
                    budget_exhaustion_stage = "fallback_preflight_guard";
                }
                fallback_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - fallback_start).count();
            }
            double search_seconds = solve_seconds - fallback_solve_seconds;
            if (search_seconds < -1e-9)
            {
                throw std::runtime_error("fallback solve timing exceeds total solve timing");
            }
            bool returned_path_current_scene_valid = false;
            bool returned_path_current_scene_validation_failure = false;
            if (solved && can_plan)
            {
                auto validation_path = planner->get_final_path();
                const auto validation_start = std::chrono::steady_clock::now();
                const bool valid_in_current_scene =
                    !validation_path.empty() && planner->check_path(validation_path);
                search_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - validation_start).count();
                returned_path_current_scene_valid = valid_in_current_scene;
                returned_path_current_scene_validation_failure =
                    !valid_in_current_scene;
                solved = solved && valid_in_current_scene;
            }
            any_returned_path_current_scene_validation_failure =
                any_returned_path_current_scene_validation_failure ||
                returned_path_current_scene_validation_failure;
            stage_timing.tube_update_ms += 1000.0 * repair.tube_update_seconds;
            stage_timing.safe_interval_update_ms +=
                1000.0 * repair.safe_interval_update_seconds;
            stage_timing.tree_repair_ms += 1000.0 * repair.tree_repair_seconds;
            stage_timing.search_ms = 1000.0 * std::max(0.0, search_seconds);
            stage_timing.fallback_ms = 1000.0 * fallback_seconds;
            const double planning_decision_wall_clock_ms =
                1000.0 * planning_elapsed_seconds();
            const double online_latency_ms = planning_decision_wall_clock_ms;
            const double unattributed_online_ms =
                stage_timing.unattributed_ms(planning_decision_wall_clock_ms);
            const bool stage_timing_valid =
                stage_timing.reconciles(planning_decision_wall_clock_ms, 0.1);
            const bool deadline_missed = rolling && online_latency_ms > deadline_ms;
            accepted_plan = solved && !deadline_missed && can_plan;
            if (returned_path_current_scene_validation_failure)
            {
                emergency_stop = true;
                accepted_plan = false;
                control_state = "safe_stop_safety_validation_failed";
                termination_reason = "safety_validation_failed";
            }
            else if (deadline_missed)
            {
                emergency_stop = true;
                accepted_plan = false;
                control_state = "safe_stop_deadline_missed";
                termination_reason = "deadline_missed";
                budget_exhaustion_stage = "total_deadline";
            }
            else if (rolling && can_plan && !solved)
            {
                emergency_stop = true;
                control_state = "safe_stop_planning_failed";
                termination_reason = "planning_failed";
            }
            else if (rolling && accepted_plan)
            {
                control_state = fallback_full_replan_used
                    ? "plan_accepted_after_full_replan_fallback"
                    : "plan_accepted";
            }
            ever_accepted_plan = ever_accepted_plan || accepted_plan;
            any_deadline_missed = any_deadline_missed || deadline_missed;
            any_prediction_unavailable = any_prediction_unavailable ||
                prediction_status == "unavailable";
            any_reliable_horizon_too_short = any_reliable_horizon_too_short ||
                reliable_horizon_too_short;
            any_repair_invariant_failure = any_repair_invariant_failure ||
                !repair.invariants_hold;
            const bool stale_prefix_executed_after_prediction_invalidation =
                prediction_status == "unavailable" &&
                (execution_advance.advanced || executed_prefix_truth.checked_frames > 0);
            any_stale_prefix_after_invalidation =
                any_stale_prefix_after_invalidation ||
                stale_prefix_executed_after_prediction_invalidation;
            if (rolling && emergency_stop)
            {
                braking = evaluate_braking_trajectory(
                    current_configuration,
                    previous_configuration,
                    braking_start_frame,
                    braking_frames,
                    truth_collision_manager,
                    truth_reader.get_scene_task());
                braking_collision = braking.truth.collision;
            }
            const auto final_path = returned_path_current_scene_validation_failure
                    ? std::vector<MDP::MSIRRT::Vertex *>{}
                    : planner->get_final_path();
            const auto truth_evaluation = evaluate_against_truth(
                final_path,
                truth_collision_manager,
                truth_reader.get_scene_task());
            const auto path_metrics = calculate_path_metrics(final_path);
            const auto prediction_coverage = evaluate_prediction_coverage(
                scene,
                truth_reader.get_scene_task(),
                static_cast<int>(scene.prediction_issue_frame),
                planner->get_planning_horizon_frame());

            rapidjson::Value item(rapidjson::kObjectType);
            item.AddMember("update_index", static_cast<uint64_t>(update_index), allocator);
            item.AddMember("prediction_file", rapidjson::Value(prediction_paths[update_index].string().c_str(), allocator), allocator);
            item.AddMember("issue_frame", static_cast<uint64_t>(scene.prediction_issue_frame), allocator);
            item.AddMember("requested_execution_frame", execution_window.requested_frame, allocator);
            item.AddMember(
                "reliable_execution_until_frame",
                execution_window.reliable_until_frame,
                allocator);
            item.AddMember("path_until_frame", execution_window.path_until_frame, allocator);
            item.AddMember("validated_execution_frame", execution_window.validated_frame, allocator);
            item.AddMember("actual_execution_frame", actual_execution_frame, allocator);
            item.AddMember("execution_prefix_frames", execution_window.prefix_frames, allocator);
            item.AddMember(
                "shortened_execution_frames",
                execution_window.shortened_frames,
                allocator);
            item.AddMember(
                "execution_limited_by_reliability",
                execution_window.limited_by_reliability,
                allocator);
            item.AddMember(
                "execution_limit_reason",
                rapidjson::Value(execution_window.limit_reason.c_str(), allocator),
                allocator);
            item.AddMember("planning_start_frame", planner->get_planning_start_frame(), allocator);
            item.AddMember("planning_horizon_frame", planner->get_planning_horizon_frame(), allocator);
            item.AddMember("maximum_reliable_horizon_frame", maximum_reliable_horizon_frame, allocator);
            item.AddMember(
                "tube_ablation_mode",
                rapidjson::Value(tube_ablation_mode.c_str(), allocator),
                allocator);
            item.AddMember(
                "fixed_tube_transformed_dynamic_spheres",
                static_cast<uint64_t>(fixed_tube_report.transformed_dynamic_spheres),
                allocator);
            item.AddMember(
                "raw_initial_planning_horizon_frame",
                raw_initial_planning_horizon_frame,
                allocator);
            item.AddMember("initial_planning_horizon_frame", initial_planning_horizon_frame, allocator);
            item.AddMember(
                "minimum_kinematic_horizon_frame",
                minimum_required_horizon_frame,
                allocator);
            item.AddMember(
                "horizon_feasibility_slack_frames",
                horizon_feasibility_slack_frames,
                allocator);
            item.AddMember(
                "skipped_infeasible_horizon_stages",
                skipped_infeasible_horizon_stages,
                allocator);
            item.AddMember(
                "stage_budget_reuse_fraction",
                stage_budget_reuse_fraction,
                allocator);
            item.AddMember(
                "saved_infeasible_stage_budget_ms",
                saved_infeasible_stage_budget_ms,
                allocator);
            item.AddMember(
                "initial_stage_budget_limit_ms",
                initial_stage_budget_limit_ms,
                allocator);
            item.AddMember(
                "progressive_guard_remaining_ms",
                progressive_guard_remaining_ms,
                allocator);
            item.AddMember("horizon_expansions", horizon_expansions, allocator);
            rapidjson::Value attempted_horizons(rapidjson::kArrayType);
            for (const int horizon : attempted_horizon_frames)
            {
                attempted_horizons.PushBack(horizon, allocator);
            }
            item.AddMember("attempted_horizon_frames", attempted_horizons, allocator);
            rapidjson::Value stage_solve_times(rapidjson::kArrayType);
            for (const double milliseconds : horizon_stage_solve_ms)
            {
                stage_solve_times.PushBack(milliseconds, allocator);
            }
            item.AddMember("horizon_stage_solve_ms", stage_solve_times, allocator);
            if (!final_path.empty())
            {
                item.AddMember(
                    "contracted_solution_horizon_frame",
                    static_cast<int>(std::ceil(final_path.back()->arrival_time)),
                    allocator);
            }
            else
            {
                item.AddMember(
                    "contracted_solution_horizon_frame",
                    rapidjson::Value().SetNull(),
                    allocator);
            }
            item.AddMember("solved", solved, allocator);
            item.AddMember("accepted_plan", accepted_plan, allocator);
            item.AddMember(
                "preserved_solution_fast_path",
                preserved_solution_fast_path,
                allocator);
            item.AddMember("solve_seconds", solve_seconds, allocator);
            item.AddMember("fallback_full_replan_used", fallback_full_replan_used, allocator);
            item.AddMember(
                "planning_budget_available_after_update_ms",
                planning_budget.available_after_update_ms,
                allocator);
            item.AddMember(
                "progressive_budget_ms", planning_budget.progressive_ms, allocator);
            item.AddMember(
                "fallback_reserved_ms",
                planning_budget.fallback_reserved_ms,
                allocator);
            item.AddMember(
                "maximum_horizon_budget_continuation_used",
                maximum_horizon_budget_continuation_used,
                allocator);
            item.AddMember(
                "final_stage_effective_guard_ms",
                final_stage_effective_guard_ms,
                allocator);
            item.AddMember(
                "final_stage_solve_budget_ms", final_stage_solve_budget_ms, allocator);
            item.AddMember(
                "final_stage_continuation_used",
                final_stage_continuation_used,
                allocator);
            item.AddMember("fallback_setup_seconds", fallback_setup_seconds, allocator);
            item.AddMember("fallback_solve_seconds", fallback_solve_seconds, allocator);
            item.AddMember(
                "fallback_preflight_solve_budget_ms",
                fallback_preflight_solve_budget_ms,
                allocator);
            item.AddMember("progressive_setup_seconds", progressive_setup_seconds, allocator);
            item.AddMember(
                "carried_initial_planner_setup_ms",
                1000.0 * carried_initial_setup_seconds,
                allocator);
            item.AddMember(
                "carried_initial_prediction_ms",
                update_index == 0 ? 1000.0 * initial_prediction_seconds : 0.0,
                allocator);
            item.AddMember(
                "carried_initial_tube_update_ms",
                update_index == 0 ? 1000.0 * initial_tube_update_seconds : 0.0,
                allocator);
            item.AddMember(
                "carried_initial_safe_interval_update_ms",
                update_index == 0 ? 1000.0 * initial_safe_interval_update_seconds : 0.0,
                allocator);
            item.AddMember(
                "excluded_truth_evaluation_ms",
                1000.0 * excluded_truth_evaluation_seconds,
                allocator);
            item.AddMember("online_latency_ms", online_latency_ms, allocator);
            item.AddMember(
                "planning_decision_wall_clock_ms",
                planning_decision_wall_clock_ms,
                allocator);
            item.AddMember("prediction_ms", stage_timing.prediction_ms, allocator);
            item.AddMember("tube_update_ms", stage_timing.tube_update_ms, allocator);
            item.AddMember(
                "safe_interval_update_ms", stage_timing.safe_interval_update_ms, allocator);
            item.AddMember("tree_repair_ms", stage_timing.tree_repair_ms, allocator);
            item.AddMember("search_ms", stage_timing.search_ms, allocator);
            item.AddMember("fallback_ms", stage_timing.fallback_ms, allocator);
            item.AddMember("unattributed_online_ms", unattributed_online_ms, allocator);
            item.AddMember("stage_timing_valid", stage_timing_valid, allocator);
            item.AddMember("deadline_missed", deadline_missed, allocator);
            item.AddMember("explicit_deadline_miss", deadline_missed, allocator);
            item.AddMember("hidden_deadline_miss", false, allocator);
            item.AddMember(
                "minimum_required_horizon_frame",
                minimum_required_horizon_frame,
                allocator);
            item.AddMember("horizon_feasible", horizon_feasible, allocator);
            item.AddMember(
                "reliable_horizon_too_short", reliable_horizon_too_short, allocator);
            item.AddMember(
                "budget_exhaustion_stage",
                rapidjson::Value(budget_exhaustion_stage.c_str(), allocator),
                allocator);
            item.AddMember("control_state", rapidjson::Value(control_state.c_str(), allocator), allocator);
            item.AddMember("prediction_version", static_cast<uint64_t>(planner->get_prediction_version()), allocator);
            item.AddMember("planner_seed", planner_seed, allocator);
            item.AddMember("start_tree_vertices", planner->get_start_tree_vertices_count(), allocator);
            item.AddMember("goal_tree_vertices", planner->get_goal_tree_vertices_count(), allocator);
            item.AddMember("start_tree_active_vertices", static_cast<uint64_t>(planner->get_start_tree_active_vertices_count()), allocator);
            item.AddMember("goal_tree_active_vertices", static_cast<uint64_t>(planner->get_goal_tree_active_vertices_count()), allocator);
            item.AddMember("evaluated_samples", static_cast<uint64_t>(planner->get_evaluated_samples() - evaluated_before), allocator);
            item.AddMember("conditionally_rejected_samples", static_cast<uint64_t>(planner->get_conditionally_rejected_samples() - rejected_before), allocator);
            item.AddMember(
                "conditional_rejected_kinematic",
                static_cast<uint64_t>(
                    planner->get_conditionally_rejected_kinematic_samples() -
                    rejected_kinematic_before),
                allocator);
            item.AddMember(
                "conditional_rejected_temporal",
                static_cast<uint64_t>(
                    planner->get_conditionally_rejected_temporal_samples() -
                    rejected_temporal_before),
                allocator);
            item.AddMember("conditionally_bypassed_samples", static_cast<uint64_t>(planner->get_conditionally_bypassed_samples() - bypassed_before), allocator);
            item.AddMember("proposed_samples", static_cast<uint64_t>(planner->get_proposed_samples() - proposed_before), allocator);
            item.AddMember(
                "conditional_exploration_base_rate",
                planner->get_conditional_exploration_base_rate(),
                allocator);
            item.AddMember(
                "conditional_exploration_effective_rate",
                planner->get_conditional_exploration_effective_rate(),
                allocator);
            item.AddMember(
                "conditional_rejection_window_ratio",
                planner->get_conditional_rejection_window_ratio(),
                allocator);
            item.AddMember(
                "conditional_rejection_feedback_triggers",
                static_cast<uint64_t>(
                    planner->get_conditional_rejection_feedback_triggers() -
                    feedback_triggers_before),
                allocator);
            item.AddMember(
                "conditional_feedback_boosted_samples",
                static_cast<uint64_t>(
                    planner->get_conditional_feedback_boosted_samples() -
                    feedback_boosted_before),
                allocator);
            item.AddMember(
                "cooperative_deadline_check_triggers",
                static_cast<uint64_t>(
                    planner->get_cooperative_deadline_check_triggers() -
                    cooperative_deadline_check_triggers_before),
                allocator);
            item.AddMember(
                "reachable_ellipsoid_samples",
                static_cast<uint64_t>(
                    planner->get_reachable_ellipsoid_samples() -
                    reachable_ellipsoid_samples_before),
                allocator);
            item.AddMember("changed_windows", frame_ranges_json(repair.changed_windows, allocator), allocator);
            item.AddMember("collision_structure_update_seconds", repair.collision_structure_update_seconds, allocator);
            item.AddMember(
                "collision_frames_rebuilt",
                static_cast<uint64_t>(repair.collision_frames_rebuilt),
                allocator);
            item.AddMember(
                "collision_obstacles_reindexed",
                static_cast<uint64_t>(repair.collision_obstacles_reindexed),
                allocator);
            item.AddMember(
                "collision_full_rebuild_fallback",
                repair.collision_full_rebuild_fallback,
                allocator);
            item.AddMember("repair_seconds", repair.repair_seconds, allocator);
            item.AddMember("tube_update_seconds", repair.tube_update_seconds, allocator);
            item.AddMember(
                "safe_interval_update_seconds",
                repair.safe_interval_update_seconds,
                allocator);
            item.AddMember("tree_repair_seconds", repair.tree_repair_seconds, allocator);
            item.AddMember(
                "used_stored_previous_intervals",
                repair.used_stored_previous_intervals,
                allocator);
            item.AddMember("previous_solution_invalidated", repair.previous_solution_invalidated, allocator);
            item.AddMember("previous_solution_preserved", repair.previous_solution_preserved, allocator);
            item.AddMember("repair_invariants_hold", repair.invariants_hold, allocator);
            item.AddMember(
                "returned_path_current_scene_valid",
                returned_path_current_scene_valid,
                allocator);
            item.AddMember(
                "returned_path_current_scene_validation_failure",
                returned_path_current_scene_validation_failure,
                allocator);
            item.AddMember("execution_advanced", execution_advance.advanced, allocator);
            item.AddMember(
                "stale_prefix_executed_after_prediction_invalidation",
                stale_prefix_executed_after_prediction_invalidation,
                allocator);
            item.AddMember(
                "execution_consumed_start_vertices",
                static_cast<uint64_t>(execution_advance.start_tree.consumed_vertices),
                allocator);
            item.AddMember(
                "execution_reused_start_vertices",
                static_cast<uint64_t>(execution_advance.start_tree.reused_vertices),
                allocator);
            item.AddMember("executed_prefix_truth_collision", executed_prefix_truth.collision, allocator);
            add_nullable_frame(
                item,
                "executed_prefix_truth_first_collision_frame",
                executed_prefix_truth.first_collision_frame,
                allocator);
            if (reactive_stop_frame >= 0)
            {
                item.AddMember("reactive_stop_frame", reactive_stop_frame, allocator);
            }
            else
            {
                item.AddMember("reactive_stop_frame", rapidjson::Value().SetNull(), allocator);
            }
            item.AddMember(
                "executed_prefix_truth_checked_frames",
                static_cast<uint64_t>(executed_prefix_truth.checked_frames),
                allocator);
            if (std::isfinite(executed_prefix_truth.min_distance))
            {
                item.AddMember(
                    "executed_prefix_truth_min_distance",
                    executed_prefix_truth.min_distance,
                    allocator);
            }
            else
            {
                item.AddMember("executed_prefix_truth_min_distance", rapidjson::Value().SetNull(), allocator);
            }
            item.AddMember("truth_collision", truth_evaluation.collision, allocator);
            add_nullable_frame(
                item,
                "truth_first_collision_frame",
                truth_evaluation.first_collision_frame,
                allocator);
            item.AddMember("truth_checked_frames", static_cast<uint64_t>(truth_evaluation.checked_frames), allocator);
            if (std::isfinite(truth_evaluation.min_distance))
            {
                item.AddMember("truth_min_distance", truth_evaluation.min_distance, allocator);
            }
            else
            {
                item.AddMember("truth_min_distance", rapidjson::Value().SetNull(), allocator);
            }
            item.AddMember("path_joint_space_length", path_metrics.joint_space_length, allocator);
            item.AddMember("path_waiting_frames", path_metrics.waiting_frames, allocator);
            if (!final_path.empty())
            {
                item.AddMember("path_arrival_frame", path_metrics.arrival_frame, allocator);
            }
            else
            {
                item.AddMember("path_arrival_frame", rapidjson::Value().SetNull(), allocator);
            }
            item.AddMember("prediction_covered_samples", static_cast<uint64_t>(prediction_coverage.covered_samples), allocator);
            item.AddMember("prediction_total_samples", static_cast<uint64_t>(prediction_coverage.total_samples), allocator);
            item.AddMember(
                "prediction_simultaneously_covered_frames",
                static_cast<uint64_t>(prediction_coverage.simultaneously_covered_frames),
                allocator);
            item.AddMember(
                "prediction_total_frames",
                static_cast<uint64_t>(prediction_coverage.total_frames),
                allocator);
            if (prediction_coverage.total_samples > 0)
            {
                item.AddMember(
                    "prediction_coverage_rate",
                    static_cast<double>(prediction_coverage.covered_samples) / prediction_coverage.total_samples,
                    allocator);
                item.AddMember(
                    "prediction_mean_center_error",
                    prediction_coverage.center_error_sum / prediction_coverage.total_samples,
                    allocator);
                item.AddMember("prediction_max_center_error", prediction_coverage.maximum_center_error, allocator);
                item.AddMember(
                    "prediction_mean_tube_radius",
                    prediction_coverage.tube_radius_sum / prediction_coverage.total_samples,
                    allocator);
                item.AddMember("prediction_max_tube_radius", prediction_coverage.maximum_tube_radius, allocator);
                item.AddMember(
                    "prediction_simultaneous_coverage_rate",
                    prediction_coverage.total_frames > 0
                        ? static_cast<double>(prediction_coverage.simultaneously_covered_frames) /
                            prediction_coverage.total_frames
                        : 0.0,
                    allocator);
            }
            else
            {
                item.AddMember("prediction_coverage_rate", rapidjson::Value().SetNull(), allocator);
                item.AddMember("prediction_mean_center_error", rapidjson::Value().SetNull(), allocator);
                item.AddMember("prediction_max_center_error", rapidjson::Value().SetNull(), allocator);
                item.AddMember("prediction_mean_tube_radius", rapidjson::Value().SetNull(), allocator);
                item.AddMember("prediction_max_tube_radius", rapidjson::Value().SetNull(), allocator);
                item.AddMember("prediction_simultaneous_coverage_rate", rapidjson::Value().SetNull(), allocator);
            }
            item.AddMember(
                "prediction_status",
                rapidjson::Value(prediction_status.c_str(), allocator),
                allocator);
            item.AddMember(
                "used_cached_prediction_scene",
                used_cached_prediction_scene,
                allocator);
            item.AddMember("emergency_stop", control_state.rfind("safe_stop", 0) == 0, allocator);
            item.AddMember("braking_truth_collision", braking.truth.collision, allocator);
            add_nullable_frame(
                item,
                "braking_truth_first_collision_frame",
                braking.truth.first_collision_frame,
                allocator);
            item.AddMember(
                "braking_truth_checked_frames",
                static_cast<uint64_t>(braking.truth.checked_frames),
                allocator);
            if (std::isfinite(braking.truth.min_distance))
            {
                item.AddMember("braking_truth_min_distance", braking.truth.min_distance, allocator);
            }
            else
            {
                item.AddMember("braking_truth_min_distance", rapidjson::Value().SetNull(), allocator);
            }
            item.AddMember("braking_trajectory", braking_trajectory_json(braking, allocator), allocator);
            item.AddMember("start_tree_repair", repair_report_json(repair.start_tree, allocator), allocator);
            item.AddMember("goal_tree_repair", repair_report_json(repair.goal_tree, allocator), allocator);
            item.AddMember("final_path", final_path_json(final_path, allocator), allocator);
            updates.PushBack(item, allocator);
            if (rolling && emergency_stop)
            {
                break;
            }
        }

        if (rolling && !emergency_stop && !task_completed)
        {
            const auto remaining_path = planner->get_final_path();
            if (!remaining_path.empty())
            {
                const auto final_execution = evaluate_against_truth(
                    remaining_path,
                    truth_collision_manager,
                    truth_reader.get_scene_task(),
                    planner->get_planning_start_frame(),
                    static_cast<int>(std::ceil(remaining_path.back()->arrival_time)),
                    reactive_stop_distance);
                merge_truth_evaluation(executed_truth, final_execution);
                if (final_execution.first_reactive_stop_frame >= 0)
                {
                    const int stop_frame = final_execution.first_reactive_stop_frame;
                    const auto stop_configuration = planner->configuration_at_frame(stop_frame);
                    const auto previous_stop_configuration = planner->configuration_at_frame(
                        std::max(planner->get_planning_start_frame(), stop_frame - 1));
                    const auto final_braking = evaluate_braking_trajectory(
                        stop_configuration,
                        previous_stop_configuration,
                        stop_frame,
                        braking_frames,
                        truth_collision_manager,
                        truth_reader.get_scene_task());
                    emergency_stop = true;
                    braking_collision = final_braking.truth.collision;
                    termination_reason = "reactive_clearance_threshold";
                }
                else
                {
                    completion_frame = remaining_path.back()->arrival_time;
                    task_completed = !final_execution.collision;
                    termination_reason = final_execution.collision
                        ? "truth_collision_before_goal"
                        : "goal_reached_after_final_update";
                }
            }
        }

        MDP::MSIRRT::FailureAttributionInput failure_attribution;
        failure_attribution.task_completed = task_completed;
        failure_attribution.safety_validation_failed =
            executed_truth.collision || braking_collision ||
            any_stale_prefix_after_invalidation || any_repair_invariant_failure ||
            any_returned_path_current_scene_validation_failure;
        failure_attribution.deadline_missed = any_deadline_missed;
        failure_attribution.prediction_unavailable = any_prediction_unavailable;
        failure_attribution.reliable_horizon_too_short =
            any_reliable_horizon_too_short;
        failure_attribution.ever_accepted_plan = ever_accepted_plan;
        failure_attribution.terminal_update_index = terminal_update_index;
        const auto primary_failure =
            MDP::MSIRRT::classify_primary_failure(failure_attribution);

        output.AddMember("updates", updates, allocator);
        output.AddMember("task_completed", task_completed, allocator);
        output.AddMember("emergency_stop", emergency_stop, allocator);
        output.AddMember("braking_collision", braking_collision, allocator);
        output.AddMember("executed_truth_collision", executed_truth.collision, allocator);
        output.AddMember(
            "stale_prefix_collision",
            any_stale_prefix_after_invalidation && executed_truth.collision,
            allocator);
        output.AddMember(
            "stale_prefix_execution_after_invalidation",
            any_stale_prefix_after_invalidation,
            allocator);
        output.AddMember(
            "tree_invariant_failure", any_repair_invariant_failure, allocator);
        output.AddMember(
            "returned_path_current_scene_valid",
            !any_returned_path_current_scene_validation_failure,
            allocator);
        output.AddMember(
            "returned_path_current_scene_validation_failure",
            any_returned_path_current_scene_validation_failure,
            allocator);
        output.AddMember("explicit_deadline_miss", any_deadline_missed, allocator);
        output.AddMember("hidden_deadline_miss", false, allocator);
        output.AddMember(
            "primary_failure_category",
            rapidjson::Value(
                MDP::MSIRRT::primary_failure_category_name(primary_failure), allocator),
            allocator);
        add_nullable_frame(
            output,
            "executed_truth_first_collision_frame",
            executed_truth.first_collision_frame,
            allocator);
        output.AddMember(
            "executed_truth_checked_frames",
            static_cast<uint64_t>(executed_truth.checked_frames),
            allocator);
        if (std::isfinite(executed_truth.min_distance))
        {
            output.AddMember("executed_truth_min_distance", executed_truth.min_distance, allocator);
        }
        else
        {
            output.AddMember("executed_truth_min_distance", rapidjson::Value().SetNull(), allocator);
        }
        if (std::isfinite(completion_frame))
        {
            output.AddMember("completion_frame", completion_frame, allocator);
        }
        else
        {
            output.AddMember("completion_frame", rapidjson::Value().SetNull(), allocator);
        }
        output.AddMember(
            "termination_reason",
            rapidjson::Value(termination_reason.c_str(), allocator),
            allocator);
        std::filesystem::create_directories(result_path.parent_path());
        std::ofstream result_stream(result_path);
        if (!result_stream)
        {
            throw std::runtime_error("cannot create result JSON: " + result_path.string());
        }
        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        output.Accept(writer);
        result_stream << buffer.GetString() << '\n';
        std::cout << result_path << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
