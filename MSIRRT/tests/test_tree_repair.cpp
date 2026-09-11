#include "Tree.hpp"

#include <cassert>
#include <iostream>

namespace {

MDP::MSIRRT::Vertex::VertexCoordType point(double value)
{
    MDP::MSIRRT::Vertex::VertexCoordType result;
    result.setConstant(value);
    return result;
}

void test_stable_ids_and_subtree_deactivation()
{
    MDP::MSIRRT::Tree tree("start", 0, 6);
    tree.add_vertex(point(0.0), {0, 100}, nullptr, -1, 0);
    auto *root = tree.array_of_vertices.back();
    tree.add_vertex(point(1.0), {0, 100}, root, 1, 2);
    auto *child = tree.array_of_vertices.back();
    tree.add_vertex(point(2.0), {0, 100}, child, 3, 4);
    auto *grandchild = tree.array_of_vertices.back();

    assert(root->vertex_id == 0);
    assert(child->vertex_id == 1);
    assert(grandchild->vertex_id == 2);
    assert(tree.active_vertex_count() == 3);

    const auto removed = tree.deactivate_subtree(child, 3);
    assert(removed.size() == 2);
    assert(root->active);
    assert(!child->active);
    assert(!grandchild->active);
    assert(child->invalidated_prediction_version == 3);
    assert(tree.active_vertex_count() == 1);

    tree.add_vertex(point(3.0), {0, 100}, root, 5, 6);
    auto *replacement = tree.array_of_vertices.back();
    assert(replacement->vertex_id == 3);
    assert(replacement->active);
    assert(tree.active_vertex_count() == 2);
}

void test_tree_temporal_dependencies_follow_lifecycle()
{
    MDP::MSIRRT::Tree tree("start", 0, 6);
    tree.set_prediction_version(4);
    tree.add_vertex(point(0.0), {0, 100}, nullptr, -1, 0);
    auto *root = tree.array_of_vertices.back();
    tree.add_vertex(point(1.0), {10, 30}, root, 12, 20);
    auto *child = tree.array_of_vertices.back();

    assert(root->prediction_version == 4);
    assert(child->prediction_version == 4);

    auto affected = tree.affected_by({MDP::MSIRRT::FrameRange(15, 15)});
    assert(affected.vertex_ids.count(root->vertex_id) == 1);
    assert(affected.vertex_ids.count(child->vertex_id) == 1);
    assert(affected.edge_ids.count(child->vertex_id) == 1);

    tree.update_vertex_interval(child, {21, 40}, 5);
    assert(child->safe_interval == std::make_pair(21, 40));
    assert(child->prediction_version == 5);
    affected = tree.affected_by({MDP::MSIRRT::FrameRange(15, 15)});
    assert(affected.vertex_ids.count(child->vertex_id) == 0);
    assert(affected.edge_ids.count(child->vertex_id) == 1);

    tree.deactivate_subtree(child, 5);
    affected = tree.affected_by({MDP::MSIRRT::FrameRange(0, 100)});
    assert(affected.vertex_ids.count(child->vertex_id) == 0);
    assert(affected.edge_ids.count(child->vertex_id) == 0);
}

void test_reroot_at_existing_vertex_consumes_only_old_prefix()
{
    MDP::MSIRRT::Tree tree("start", 0, 6);
    tree.add_vertex(point(0.0), {0, 100}, nullptr, -1, 0);
    auto *root = tree.array_of_vertices.back();
    tree.add_vertex(point(1.0), {0, 100}, root, 1, 2);
    auto *path_child = tree.array_of_vertices.back();
    tree.add_vertex(point(2.0), {0, 100}, path_child, 3, 4);
    auto *suffix = tree.array_of_vertices.back();
    tree.add_vertex(point(-1.0), {0, 100}, root, 1, 2);
    auto *side_branch = tree.array_of_vertices.back();

    const auto report = tree.reroot_at_vertex(path_child, {3, 100}, 3, 2);

    assert(report.root == path_child);
    assert(report.active_vertices_before == 4);
    assert(report.reused_vertices == 2);
    assert(report.consumed_vertices == 2);
    assert(report.invariants_hold);
    assert(!root->active);
    assert(path_child->active && path_child->parent == nullptr);
    assert(path_child->arrival_time == 3);
    assert(suffix->active && suffix->parent == path_child);
    assert(!side_branch->active);
}

void test_reroot_inside_edge_inserts_root_and_preserves_suffix_ids()
{
    MDP::MSIRRT::Tree tree("start", 0, 6);
    tree.add_vertex(point(0.0), {0, 100}, nullptr, -1, 0);
    auto *root = tree.array_of_vertices.back();
    tree.add_vertex(point(1.0), {0, 100}, root, 1, 2);
    auto *path_child = tree.array_of_vertices.back();
    tree.add_vertex(point(2.0), {0, 100}, path_child, 3, 4);
    auto *suffix = tree.array_of_vertices.back();
    tree.add_vertex(point(-1.0), {0, 100}, root, 1, 2);

    const auto report = tree.reroot_on_edge(path_child, point(0.5), {1, 100}, 1, 3);

    assert(report.root->vertex_id == 4);
    assert(report.active_vertices_before == 4);
    assert(report.reused_vertices == 2);
    assert(report.consumed_vertices == 2);
    assert(report.invariants_hold);
    assert(!root->active);
    assert(path_child->active && path_child->vertex_id == 1);
    assert(path_child->parent == report.root);
    assert(path_child->departure_from_parent_time == 1);
    assert(suffix->active && suffix->vertex_id == 2);
    assert(tree.active_vertex_count() == 3);
}

} // namespace

int main()
{
    test_stable_ids_and_subtree_deactivation();
    test_tree_temporal_dependencies_follow_lifecycle();
    test_reroot_at_existing_vertex_consumes_only_old_prefix();
    test_reroot_inside_edge_inserts_root_and_preserves_suffix_ids();
    std::cout << "tree repair tests passed\n";
    return 0;
}
