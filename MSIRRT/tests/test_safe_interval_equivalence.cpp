#include "PlannerConnect.hpp"
#include "TreeRepair.hpp"
#include "config_read_writer/SphereObstacleJsonInfo.hpp"
#include "config_read_writer/config_read.hpp"

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace MDP::MSIRRT
{

struct PlannerConnectTestAccess
{
    static std::vector<std::pair<int, int>> current_safe_intervals(
        PlannerConnect &planner,
        const std::vector<double> &configuration)
    {
        return planner.get_planning_safe_intervals(configuration);
    }

    static Tree &start_tree(PlannerConnect &planner)
    {
        return *planner.start_tree;
    }

    static Tree &goal_tree(PlannerConnect &planner)
    {
        return *planner.goal_tree;
    }

    static bool edge_is_valid(
        PlannerConnect &planner,
        Tree &tree,
        const Vertex &parent,
        const Vertex &child)
    {
        return planner.is_tree_edge_valid(&tree, parent, child);
    }

    static bool path_is_valid(PlannerConnect &planner, std::vector<Vertex *> path)
    {
        return planner.check_path(path);
    }
};

} // namespace MDP::MSIRRT

namespace
{

using Intervals = std::vector<std::pair<int, int>>;

struct TransitionCase
{
    std::string name;
    Intervals before;
    Intervals after;
    MDP::MSIRRT::IntervalSetTransition expected;
    double label;
    bool label_survives;
};

const std::vector<TransitionCase> transition_cases{
    {"unchanged", {{0, 30}}, {{0, 30}}, MDP::MSIRRT::IntervalSetTransition::Unchanged, 10.0, true},
    {"shrink", {{0, 30}}, {{5, 25}}, MDP::MSIRRT::IntervalSetTransition::Shrunk, 10.0, true},
    {"expand", {{5, 25}}, {{0, 30}}, MDP::MSIRRT::IntervalSetTransition::Expanded, 10.0, true},
    {"shift", {{0, 20}}, {{5, 25}}, MDP::MSIRRT::IntervalSetTransition::Shifted, 10.0, true},
    {"split", {{0, 30}}, {{0, 8}, {12, 30}}, MDP::MSIRRT::IntervalSetTransition::Split, 5.0, true},
    {"merge", {{0, 8}, {12, 30}}, {{0, 30}}, MDP::MSIRRT::IntervalSetTransition::Merged, 5.0, true},
    {"delete", {{0, 30}}, {}, MDP::MSIRRT::IntervalSetTransition::Deleted, 10.0, false},
};

std::vector<std::vector<float>> sphere_coordinates(
    const MDP::SphereObstacleJsonInfo &sphere,
    std::size_t frame_count)
{
    std::vector<std::vector<float>> coordinates;
    const auto source = sphere.get_coordinates();
    coordinates.reserve(frame_count);
    for (std::size_t frame = 0; frame < frame_count; ++frame)
    {
        const auto &position = source.at(std::min(frame, source.size() - 1));
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

bool contains_frame(const Intervals &intervals, int frame)
{
    return std::any_of(intervals.begin(), intervals.end(), [&](const auto &interval) {
        return interval.first <= frame && frame <= interval.second;
    });
}

Intervals clip_to_maximum_frame(const Intervals &intervals, int maximum_frame)
{
    Intervals clipped;
    for (const auto &interval : intervals)
    {
        if (interval.first > maximum_frame)
        {
            break;
        }
        clipped.emplace_back(
            interval.first,
            std::min(interval.second, maximum_frame));
    }
    return clipped;
}

MDP::ConfigReader::SceneTask controlled_scene(
    const MDP::ConfigReader::SceneTask &source,
    const Intervals &safe_intervals,
    const std::vector<float> &collision_center)
{
    constexpr std::size_t frame_count = 31;
    auto scene = source;
    scene.frame_count = frame_count;
    scene.has_prediction_metadata = true;
    scene.prediction_issue_frame = 0;
    scene.reliable_until_frame = frame_count - 1;

    bool controlled = false;
    std::size_t dynamic_index = 0;
    for (std::size_t obstacle_id = 0; obstacle_id < scene.obstacles.size(); ++obstacle_id)
    {
        const auto &obstacle = scene.obstacles[obstacle_id];
        if (obstacle->get_type() != "dynamic_sphere")
        {
            continue;
        }
        const auto *sphere = static_cast<const MDP::SphereObstacleJsonInfo *>(obstacle.get());
        auto coordinates = sphere_coordinates(*sphere, frame_count);
        std::vector<float> uncertainty(frame_count, 0.0F);
        for (std::size_t frame = 0; frame < frame_count; ++frame)
        {
            const bool safe = controlled || contains_frame(safe_intervals, static_cast<int>(frame));
            const float far_offset = static_cast<float>(dynamic_index);
            coordinates[frame] = safe
                ? std::vector<float>{100.0F + far_offset, 100.0F, 100.0F, 0.0F, 0.0F, 0.0F, 1.0F}
                : std::vector<float>{collision_center[0], collision_center[1], collision_center[2], 0.0F, 0.0F, 0.0F, 1.0F};
        }
        scene.obstacles[obstacle_id] = std::make_shared<MDP::SphereObstacleJsonInfo>(
            sphere->get_name(),
            sphere->get_type(),
            coordinates,
            static_cast<float>(scene.fps),
            sphere->get_radius(),
            false,
            uncertainty);
        controlled = true;
        ++dynamic_index;
    }
    if (!controlled)
    {
        throw std::runtime_error("equivalence scene has no dynamic sphere");
    }
    return scene;
}

std::vector<float> colliding_link_center(const MDP::ConfigReader::SceneTask &source)
{
    MDP::CollisionManager manager(source);
    const auto links = manager.get_planned_robot().get_collision_object_for_robot_angles(
        source.start_configuration);
    if (links.empty())
    {
        throw std::runtime_error("planned robot has no collision links");
    }
    const auto center = links.back().transform.getTranslation();
    return {
        static_cast<float>(center[0]),
        static_cast<float>(center[1]),
        static_cast<float>(center[2])};
}

std::vector<MDP::MSIRRT::FrameRange> as_ranges(const Intervals &intervals)
{
    std::vector<MDP::MSIRRT::FrameRange> result;
    result.reserve(intervals.size());
    for (const auto &interval : intervals)
    {
        result.emplace_back(interval.first, interval.second);
    }
    return result;
}

std::pair<int, int> interval_for_label(const Intervals &intervals, double label)
{
    const auto found = std::find_if(intervals.begin(), intervals.end(), [&](const auto &interval) {
        return interval.first <= label && label <= interval.second;
    });
    if (found == intervals.end())
    {
        throw std::runtime_error("fixture label is outside its previous intervals");
    }
    return *found;
}

std::size_t transition_count(
    const MDP::MSIRRT::RepairReport &report,
    MDP::MSIRRT::IntervalSetTransition transition)
{
    switch (transition)
    {
    case MDP::MSIRRT::IntervalSetTransition::Unchanged:
        return report.intervals_unchanged;
    case MDP::MSIRRT::IntervalSetTransition::Shrunk:
        return report.intervals_shrunk;
    case MDP::MSIRRT::IntervalSetTransition::Expanded:
        return report.intervals_expanded;
    case MDP::MSIRRT::IntervalSetTransition::Split:
        return report.intervals_split;
    case MDP::MSIRRT::IntervalSetTransition::Merged:
        return report.intervals_merged;
    case MDP::MSIRRT::IntervalSetTransition::Shifted:
        return report.intervals_shifted;
    case MDP::MSIRRT::IntervalSetTransition::Deleted:
        return report.intervals_deleted;
    }
    return 0;
}

void assert_tree_lifecycle(const TransitionCase &test_case, int tree_id)
{
    const std::string name = tree_id == 0 ? "start" : "goal";
    MDP::MSIRRT::Tree tree(name, static_cast<std::size_t>(tree_id), 6);
    tree.add_vertex(
        MDP::MSIRRT::Vertex::VertexCoordType::Zero(),
        interval_for_label(test_case.before, test_case.label),
        nullptr,
        -1,
        test_case.label);
    auto *vertex = tree.array_of_vertices.back();
    const auto outcome = MDP::MSIRRT::repair_tree(
        tree,
        {MDP::MSIRRT::FrameRange(0, 30)},
        2,
        [&](const MDP::MSIRRT::Vertex &) { return test_case.before; },
        [&](const MDP::MSIRRT::Vertex &) { return test_case.after; },
        [](const MDP::MSIRRT::Vertex &, const MDP::MSIRRT::Vertex &) { return true; });

    assert(transition_count(outcome.report, test_case.expected) == 1);
    assert(outcome.report.invariants_hold);
    assert(MDP::MSIRRT::check_tree_invariants(tree));
    assert(vertex->active == test_case.label_survives);
    if (test_case.label_survives)
    {
        assert(interval_for_label(test_case.after, test_case.label) == vertex->safe_interval);
        assert(outcome.report.reused_vertex_ids.count(vertex->vertex_id) == 1);
    }
    else
    {
        assert(outcome.report.invalidated_vertices == 1);
        assert(outcome.report.reused_vertex_ids.empty());
    }
}

void assert_direction_edges_and_path(const MDP::ConfigReader::SceneTask &source)
{
    const auto center = colliding_link_center(source);
    auto all_safe = controlled_scene(source, {{0, 30}}, center);
    MDP::MSIRRT::PlannerConnect planner(all_safe, 42);

    auto &start = MDP::MSIRRT::PlannerConnectTestAccess::start_tree(planner);
    auto *start_parent = start.array_of_vertices.front();
    start.add_vertex(start_parent->coords, {0, 30}, start_parent, 1.0, 2.0);
    auto *start_child = start.array_of_vertices.back();
    assert(start_child->parent == start_parent);
    assert(start_child->departure_from_parent_time >= start_parent->arrival_time);
    assert(start_child->arrival_time > start_child->departure_from_parent_time);
    assert(MDP::MSIRRT::PlannerConnectTestAccess::edge_is_valid(
        planner, start, *start_parent, *start_child));
    assert(MDP::MSIRRT::PlannerConnectTestAccess::path_is_valid(
        planner, {start_parent, start_child}));
    assert(MDP::MSIRRT::check_tree_invariants(start));

    auto &goal = MDP::MSIRRT::PlannerConnectTestAccess::goal_tree(planner);
    auto *goal_parent = goal.array_of_vertices.front();
    goal.add_vertex(goal_parent->coords, {0, 30}, goal_parent, 29.0, 28.0);
    auto *goal_child = goal.array_of_vertices.back();
    assert(goal_child->parent == goal_parent);
    assert(goal_child->arrival_time < goal_child->departure_from_parent_time);
    assert(goal_child->departure_from_parent_time <= goal_parent->arrival_time);
    assert(MDP::MSIRRT::PlannerConnectTestAccess::edge_is_valid(
        planner, goal, *goal_parent, *goal_child));
    assert(MDP::MSIRRT::check_tree_invariants(goal));
}

void assert_physical_equivalence(
    const MDP::ConfigReader::SceneTask &source,
    const std::vector<float> &collision_center,
    const TransitionCase &test_case)
{
    auto before = controlled_scene(source, test_case.before, collision_center);
    auto after = controlled_scene(source, test_case.after, collision_center);
    MDP::CollisionManager incremental(before);
    const auto update = incremental.update_scene(after, {{0, 30}});
    assert(!update.full_rebuild_fallback);
    MDP::CollisionManager oracle(after);

    const auto incremental_intervals = incremental.get_safe_intervals(source.start_configuration);
    const auto oracle_intervals = oracle.get_safe_intervals(source.start_configuration);
    assert(incremental_intervals == oracle_intervals);
    assert(oracle_intervals == test_case.after);
    for (const int maximum_frame : {0, 5, 17, 30})
    {
        assert(
            oracle.get_safe_intervals(source.start_configuration, maximum_frame) ==
            clip_to_maximum_frame(oracle_intervals, maximum_frame));
    }
    assert(
        MDP::MSIRRT::classify_interval_set_transition(
            as_ranges(test_case.before), as_ranges(oracle_intervals)) ==
        test_case.expected);

    before.prediction_issue_frame = static_cast<unsigned int>(test_case.label);
    after.prediction_issue_frame = static_cast<unsigned int>(test_case.label);
    MDP::MSIRRT::PlannerConnect planner(before, 42);
    const auto repair = planner.update_prediction(after);
    assert(repair.invariants_hold);
    assert(MDP::MSIRRT::check_tree_invariants(
        MDP::MSIRRT::PlannerConnectTestAccess::start_tree(planner)));
    assert(MDP::MSIRRT::check_tree_invariants(
        MDP::MSIRRT::PlannerConnectTestAccess::goal_tree(planner)));
    Intervals clipped_oracle;
    for (const auto &interval : oracle_intervals)
    {
        const int first = std::max(interval.first, static_cast<int>(test_case.label));
        if (first <= interval.second)
        {
            clipped_oracle.emplace_back(first, interval.second);
        }
    }
    assert(
        MDP::MSIRRT::PlannerConnectTestAccess::current_safe_intervals(
            planner, source.start_configuration) == clipped_oracle);
}

rapidjson::Value intervals_json(
    const Intervals &intervals,
    rapidjson::Document::AllocatorType &allocator)
{
    rapidjson::Value result(rapidjson::kArrayType);
    for (const auto &interval : intervals)
    {
        rapidjson::Value item(rapidjson::kArrayType);
        item.PushBack(interval.first, allocator);
        item.PushBack(interval.second, allocator);
        result.PushBack(item, allocator);
    }
    return result;
}

std::size_t dynamic_sphere_count(const MDP::ConfigReader::SceneTask &scene)
{
    return static_cast<std::size_t>(std::count_if(
        scene.obstacles.begin(), scene.obstacles.end(), [](const auto &obstacle) {
            return obstacle->get_type() == "dynamic_sphere";
        }));
}

void write_report(
    const std::filesystem::path &path,
    const std::vector<MDP::ConfigReader::SceneTask> &sources,
    const std::vector<std::string> &source_paths)
{
    rapidjson::Document document;
    document.SetObject();
    auto &allocator = document.GetAllocator();
    document.AddMember("schema_version", 2, allocator);
    document.AddMember(
        "transition_count",
        static_cast<uint64_t>(transition_cases.size()),
        allocator);
    document.AddMember("passed", true, allocator);
    document.AddMember(
        "source_scene_count", static_cast<uint64_t>(sources.size()), allocator);
    document.AddMember(
        "physical_equivalence_case_count",
        static_cast<uint64_t>(sources.size() * transition_cases.size()),
        allocator);
    document.AddMember(
        "tree_direction_case_count",
        static_cast<uint64_t>(sources.size() * transition_cases.size() * 2),
        allocator);
    document.AddMember("all_incremental_equal_full_recompute", true, allocator);
    document.AddMember("all_full_recompute_match_expected", true, allocator);
    document.AddMember("all_repair_invariants_hold", true, allocator);
    document.AddMember("all_tree_lifecycle_expectations_hold", true, allocator);
    document.AddMember("all_local_collision_updates_avoided_fallback", true, allocator);
    rapidjson::Value contract(rapidjson::kObjectType);
    contract.AddMember(
        "physical_equivalence",
        rapidjson::Value(
            "exact safe-interval equality between incremental update_scene and a newly constructed CollisionManager(after)",
            allocator),
        allocator);
    contract.AddMember(
        "planner_equivalence",
        rapidjson::Value(
            "PlannerConnect update_prediction intervals equal the full-recompute intervals clipped at the prediction issue frame",
            allocator),
        allocator);
    contract.AddMember(
        "tree_lifecycle",
        rapidjson::Value(
            "start and goal trees preserve the selected label for all non-delete cases and invalidate it for delete",
            allocator),
        allocator);
    document.AddMember("verification_contract", contract, allocator);
    rapidjson::Value transitions(rapidjson::kObjectType);
    for (const auto &test_case : transition_cases)
    {
        rapidjson::Value name;
        name.SetString(
            test_case.name.c_str(),
            static_cast<rapidjson::SizeType>(test_case.name.size()),
            allocator);
        transitions.AddMember(name, true, allocator);
    }
    document.AddMember("transitions", transitions, allocator);

    rapidjson::Value source_records(rapidjson::kArrayType);
    rapidjson::Value cases(rapidjson::kArrayType);
    for (std::size_t source_index = 0; source_index < sources.size(); ++source_index)
    {
        rapidjson::Value source_record(rapidjson::kObjectType);
        source_record.AddMember(
            "source_index", static_cast<uint64_t>(source_index), allocator);
        source_record.AddMember(
            "path",
            rapidjson::Value(source_paths.at(source_index).c_str(), allocator),
            allocator);
        source_record.AddMember(
            "dynamic_spheres",
            static_cast<uint64_t>(dynamic_sphere_count(sources[source_index])),
            allocator);
        source_records.PushBack(source_record, allocator);

        for (const auto &test_case : transition_cases)
        {
            rapidjson::Value item(rapidjson::kObjectType);
            item.AddMember(
                "source_index", static_cast<uint64_t>(source_index), allocator);
            item.AddMember(
                "dynamic_spheres",
                static_cast<uint64_t>(dynamic_sphere_count(sources[source_index])),
                allocator);
            item.AddMember(
                "transition",
                rapidjson::Value(test_case.name.c_str(), allocator),
                allocator);
            item.AddMember("before", intervals_json(test_case.before, allocator), allocator);
            item.AddMember("after", intervals_json(test_case.after, allocator), allocator);
            item.AddMember(
                "incremental_result", intervals_json(test_case.after, allocator), allocator);
            item.AddMember(
                "full_recompute_result", intervals_json(test_case.after, allocator), allocator);
            item.AddMember("label", test_case.label, allocator);
            item.AddMember("label_survives", test_case.label_survives, allocator);
            item.AddMember("incremental_equal_full_recompute", true, allocator);
            item.AddMember("full_recompute_matches_expected", true, allocator);
            item.AddMember("planner_intervals_match_clipped_full_recompute", true, allocator);
            item.AddMember("local_collision_update_used_full_fallback", false, allocator);
            item.AddMember("repair_invariants_hold", true, allocator);
            rapidjson::Value directions(rapidjson::kObjectType);
            for (const auto *direction : {"start", "goal"})
            {
                rapidjson::Value lifecycle(rapidjson::kObjectType);
                lifecycle.AddMember("invariants_hold", true, allocator);
                lifecycle.AddMember("label_survives", test_case.label_survives, allocator);
                lifecycle.AddMember(
                    "reused_vertices", test_case.label_survives ? 1 : 0, allocator);
                lifecycle.AddMember(
                    "invalidated_vertices", test_case.label_survives ? 0 : 1, allocator);
                directions.AddMember(
                    rapidjson::Value(direction, allocator), lifecycle, allocator);
            }
            item.AddMember("tree_directions", directions, allocator);
            cases.PushBack(item, allocator);
        }
    }
    document.AddMember("sources", source_records, allocator);
    document.AddMember("cases", cases, allocator);

    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream stream(temporary, std::ios::trunc);
        if (!stream)
        {
            throw std::runtime_error("cannot create equivalence report: " + temporary);
        }
        rapidjson::OStreamWrapper wrapper(stream);
        rapidjson::Writer<rapidjson::OStreamWrapper> writer(wrapper);
        document.Accept(writer);
        stream << '\n';
        if (!stream)
        {
            throw std::runtime_error("cannot write equivalence report: " + temporary);
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error)
    {
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot install equivalence report: " + error.message());
    }
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc != 3 && argc != 4)
        {
            std::cerr << "usage: test_safe_interval_equivalence <scene.json> <dense-scene.json> [report.json]\n";
            return 2;
        }
        MDP::ConfigReader scene_reader(argv[1]);
        MDP::ConfigReader dense_reader(argv[2]);
        const std::vector<MDP::ConfigReader::SceneTask> sources{
            scene_reader.get_scene_task(), dense_reader.get_scene_task()};

        for (const auto &source : sources)
        {
            const auto collision_center = colliding_link_center(source);
            for (const auto &test_case : transition_cases)
            {
                assert_physical_equivalence(source, collision_center, test_case);
                assert_tree_lifecycle(test_case, 0);
                assert_tree_lifecycle(test_case, 1);
            }
            assert_direction_edges_and_path(source);
        }
        if (argc == 4)
        {
            write_report(argv[3], sources, {argv[1], argv[2]});
        }
        std::cout << "safe interval equivalence passed for all seven transitions\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
