#include "TreeRepair.hpp"

#include <cassert>
#include <iostream>
#include <unordered_map>

namespace {

MDP::MSIRRT::Vertex::VertexCoordType point(double value)
{
    MDP::MSIRRT::Vertex::VertexCoordType result;
    result.setConstant(value);
    return result;
}

struct TransitionFixtureResult
{
    MDP::MSIRRT::TreeRepairOutcome outcome;
    bool active = false;
    std::pair<int, int> selected_interval{0, -1};
};

TransitionFixtureResult run_interval_transition_fixture(
    const std::vector<std::pair<int, int>> &before,
    const std::vector<std::pair<int, int>> &after,
    double arrival_time)
{
    MDP::MSIRRT::Tree tree("start", 0, 6);
    const auto stored = *std::find_if(before.begin(), before.end(), [&](const auto &interval) {
        return interval.first <= arrival_time && arrival_time <= interval.second;
    });
    tree.add_vertex(point(0.0), stored, nullptr, -1, arrival_time);
    auto *vertex = tree.array_of_vertices.back();
    auto outcome = MDP::MSIRRT::repair_tree(
        tree,
        {MDP::MSIRRT::FrameRange(0, 50)},
        2,
        [&](const MDP::MSIRRT::Vertex &) { return before; },
        [&](const MDP::MSIRRT::Vertex &) { return after; },
        [](const MDP::MSIRRT::Vertex &, const MDP::MSIRRT::Vertex &) { return true; });
    return {std::move(outcome), vertex->active, vertex->safe_interval};
}

void test_exact_interval_set_transition_counters()
{
    const auto split = run_interval_transition_fixture(
        {{0, 30}}, {{0, 5}, {10, 30}}, 2.0);
    assert(split.outcome.report.intervals_split == 1);
    assert(split.active && split.selected_interval == std::make_pair(0, 5));
    assert(split.outcome.report.reused_vertex_ids.count(0) == 1);
    assert(split.outcome.report.invariants_hold);

    const auto merged = run_interval_transition_fixture(
        {{0, 5}, {10, 30}}, {{0, 30}}, 2.0);
    assert(merged.outcome.report.intervals_merged == 1);
    assert(merged.active && merged.selected_interval == std::make_pair(0, 30));
    assert(merged.outcome.report.reused_vertex_ids.count(0) == 1);
    assert(merged.outcome.report.invariants_hold);

    const auto shifted = run_interval_transition_fixture(
        {{0, 10}}, {{5, 15}}, 7.0);
    assert(shifted.outcome.report.intervals_shifted == 1);
    assert(shifted.active && shifted.selected_interval == std::make_pair(5, 15));
    assert(shifted.outcome.report.reused_vertex_ids.count(0) == 1);
    assert(shifted.outcome.report.invariants_hold);

    const auto shrunk = run_interval_transition_fixture(
        {{0, 10}}, {{2, 8}}, 5.0);
    assert(shrunk.outcome.report.intervals_shrunk == 1);
    assert(shrunk.active && shrunk.selected_interval == std::make_pair(2, 8));
    assert(shrunk.outcome.report.reused_vertex_ids.count(0) == 1);
    assert(shrunk.outcome.report.invariants_hold);

    const auto expanded = run_interval_transition_fixture(
        {{2, 8}}, {{0, 10}}, 5.0);
    assert(expanded.outcome.report.intervals_expanded == 1);
    assert(expanded.active && expanded.selected_interval == std::make_pair(0, 10));
    assert(expanded.outcome.report.reused_vertex_ids.count(0) == 1);
    assert(expanded.outcome.report.invariants_hold);

    const auto deleted = run_interval_transition_fixture(
        {{0, 10}}, {}, 5.0);
    assert(deleted.outcome.report.intervals_deleted == 1);
    assert(!deleted.active);
    assert(deleted.outcome.report.invalidated_vertices == 1);
    assert(deleted.outcome.report.reused_vertex_ids.empty());
    assert(deleted.outcome.report.invariants_hold);
}

void test_interval_split_preserves_matching_label()
{
    MDP::MSIRRT::Tree tree("start", 0, 6);
    tree.add_vertex(point(0.0), {0, 30}, nullptr, -1, 0);
    auto *root = tree.array_of_vertices.back();
    tree.add_vertex(point(1.0), {0, 30}, root, 10, 12);
    auto *child = tree.array_of_vertices.back();

    int interval_queries = 0;
    const auto outcome = MDP::MSIRRT::repair_tree(
        tree,
        {MDP::MSIRRT::FrameRange(10, 15)},
        2,
        [&](const MDP::MSIRRT::Vertex &vertex) {
            return std::vector<std::pair<int, int>>{vertex.safe_interval};
        },
        [&](const MDP::MSIRRT::Vertex &vertex) {
            ++interval_queries;
            if (vertex.vertex_id == child->vertex_id)
            {
                return std::vector<std::pair<int, int>>{{0, 5}, {10, 15}, {20, 30}};
            }
            return std::vector<std::pair<int, int>>{{0, 30}};
        },
        [](const MDP::MSIRRT::Vertex &, const MDP::MSIRRT::Vertex &) { return true; });

    assert(interval_queries == 2);
    assert(child->active);
    assert(child->safe_interval == std::make_pair(10, 15));
    assert(outcome.report.invalidated_vertices == 0);
    assert(outcome.report.reused_vertices == 2);
    assert(outcome.report.intervals_split == 1);
    assert(outcome.report.intervals_unchanged == 1);
    assert(outcome.report.invariants_hold);
}

void test_missing_label_interval_invalidates_subtree()
{
    MDP::MSIRRT::Tree tree("start", 0, 6);
    tree.add_vertex(point(0.0), {0, 50}, nullptr, -1, 0);
    auto *root = tree.array_of_vertices.back();
    tree.add_vertex(point(1.0), {10, 30}, root, 10, 12);
    auto *child = tree.array_of_vertices.back();
    tree.add_vertex(point(2.0), {10, 40}, child, 13, 20);
    auto *grandchild = tree.array_of_vertices.back();

    const auto outcome = MDP::MSIRRT::repair_tree(
        tree,
        {MDP::MSIRRT::FrameRange(10, 30)},
        3,
        [&](const MDP::MSIRRT::Vertex &vertex) {
            return std::vector<std::pair<int, int>>{vertex.safe_interval};
        },
        [&](const MDP::MSIRRT::Vertex &vertex) {
            if (vertex.vertex_id == child->vertex_id)
            {
                return std::vector<std::pair<int, int>>{{20, 30}};
            }
            return std::vector<std::pair<int, int>>{{0, 50}};
        },
        [](const MDP::MSIRRT::Vertex &, const MDP::MSIRRT::Vertex &) { return true; });

    assert(root->active);
    assert(!child->active);
    assert(!grandchild->active);
    assert(outcome.invalidated_vertices.size() == 2);
    assert(outcome.report.invalidated_vertices == 2);
    assert(outcome.report.reused_vertices == 1);
    assert(outcome.report.intervals_shrunk == 1);
    assert(outcome.report.intervals_deleted == 0);
}

void test_invalid_edge_invalidates_only_its_branch()
{
    MDP::MSIRRT::Tree tree("goal", 1, 6);
    tree.add_vertex(point(0.0), {0, 50}, nullptr, -1, 50);
    auto *root = tree.array_of_vertices.back();
    tree.add_vertex(point(1.0), {0, 50}, root, 42, 40);
    auto *left = tree.array_of_vertices.back();
    tree.add_vertex(point(2.0), {0, 50}, root, 38, 35);
    auto *right = tree.array_of_vertices.back();

    const auto outcome = MDP::MSIRRT::repair_tree(
        tree,
        {MDP::MSIRRT::FrameRange(36, 43)},
        7,
        [](const MDP::MSIRRT::Vertex &vertex) {
            return std::vector<std::pair<int, int>>{vertex.safe_interval};
        },
        [](const MDP::MSIRRT::Vertex &) {
            return std::vector<std::pair<int, int>>{{0, 50}};
        },
        [&](const MDP::MSIRRT::Vertex &, const MDP::MSIRRT::Vertex &child) {
            return child.vertex_id != left->vertex_id;
        });

    assert(root->active);
    assert(!left->active);
    assert(right->active);
    assert(outcome.report.revalidated_edges == 2);
    assert(outcome.report.invalidated_vertices == 1);
}

void test_fractional_label_below_interval_is_not_rounded_into_safety()
{
    MDP::MSIRRT::Tree tree("goal", 1, 6);
    tree.add_vertex(point(0.0), {0, 50}, nullptr, -1, 50);
    auto *root = tree.array_of_vertices.back();
    tree.add_vertex(point(1.0), {0, 50}, root, 40, 39.8);
    auto *child = tree.array_of_vertices.back();

    const auto outcome = MDP::MSIRRT::repair_tree(
        tree,
        {MDP::MSIRRT::FrameRange(40, 50)},
        8,
        [](const MDP::MSIRRT::Vertex &vertex) {
            return std::vector<std::pair<int, int>>{vertex.safe_interval};
        },
        [&](const MDP::MSIRRT::Vertex &vertex) {
            if (vertex.vertex_id == child->vertex_id)
            {
                return std::vector<std::pair<int, int>>{{40, 50}};
            }
            return std::vector<std::pair<int, int>>{{0, 50}};
        },
        [](const MDP::MSIRRT::Vertex &, const MDP::MSIRRT::Vertex &) { return true; });

    assert(!child->active);
    assert(outcome.report.invalidated_vertices == 1);
    assert(outcome.report.invariants_hold);
}

void test_missing_previous_query_reuses_stored_selected_interval()
{
    MDP::MSIRRT::Tree tree("start", 0, 6);
    tree.add_vertex(point(0.0), {0, 30}, nullptr, -1, 2);
    auto *root = tree.array_of_vertices.back();

    int current_interval_queries = 0;
    const auto outcome = MDP::MSIRRT::repair_tree(
        tree,
        {MDP::MSIRRT::FrameRange(6, 9)},
        9,
        MDP::MSIRRT::SafeIntervalQuery{},
        [&](const MDP::MSIRRT::Vertex &) {
            ++current_interval_queries;
            return std::vector<std::pair<int, int>>{{0, 5}, {10, 30}};
        },
        [](const MDP::MSIRRT::Vertex &, const MDP::MSIRRT::Vertex &) { return true; });

    assert(current_interval_queries == 1);
    assert(root->active);
    assert(root->safe_interval == std::make_pair(0, 5));
    assert(outcome.report.intervals_split == 1);
    assert(outcome.report.reused_vertices == 1);
    assert(outcome.report.invariants_hold);
}

} // namespace

int main()
{
    test_exact_interval_set_transition_counters();
    test_interval_split_preserves_matching_label();
    test_missing_label_interval_invalidates_subtree();
    test_invalid_edge_invalidates_only_its_branch();
    test_fractional_label_below_interval_is_not_rounded_into_safety();
    test_missing_previous_query_reuses_stored_selected_interval();
    std::cout << "tree repair engine tests passed\n";
    return 0;
}
