#include "TreeRepair.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>

namespace MDP::MSIRRT
{
namespace
{

bool has_selected_ancestor(const Vertex *vertex, const std::unordered_set<std::size_t> &selected)
{
    const Vertex *ancestor = vertex->parent;
    while (ancestor != nullptr)
    {
        if (selected.count(ancestor->vertex_id) != 0)
        {
            return true;
        }
        ancestor = ancestor->parent;
    }
    return false;
}

std::vector<FrameRange> as_frame_ranges(const std::vector<std::pair<int, int>> &intervals)
{
    std::vector<FrameRange> result;
    result.reserve(intervals.size());
    for (const auto &interval : intervals)
    {
        result.emplace_back(interval.first, interval.second);
    }
    return normalize_frame_ranges(result);
}

} // namespace

TreeRepairOutcome repair_tree(
    Tree &tree,
    const std::vector<FrameRange> &changed_windows,
    std::size_t prediction_version,
    const SafeIntervalQuery &previous_safe_interval_query,
    const SafeIntervalQuery &current_safe_interval_query,
    const EdgeValidation &edge_validation)
{
    TreeRepairOutcome outcome;
    outcome.report.active_vertices_before = tree.active_vertex_count();
    std::unordered_set<std::size_t> active_vertex_ids_before;
    for (const Vertex *vertex : tree.array_of_vertices)
    {
        if (vertex->active)
        {
            active_vertex_ids_before.insert(vertex->vertex_id);
        }
    }
    tree.set_prediction_version(prediction_version);

    const AffectedDependencies affected = tree.affected_by(changed_windows);
    outcome.report.candidate_vertices = affected.vertex_ids.size();
    outcome.report.candidate_edges = affected.edge_ids.size();

    std::unordered_set<std::size_t> invalid_roots;
    for (const std::size_t vertex_id : affected.vertex_ids)
    {
        if (vertex_id >= tree.array_of_vertices.size())
        {
            continue;
        }
        Vertex *vertex = tree.array_of_vertices[vertex_id];
        if (!vertex->active)
        {
            continue;
        }

        ++outcome.report.revalidated_vertices;
        const auto previous_intervals = previous_safe_interval_query
            ? as_frame_ranges(previous_safe_interval_query(*vertex))
            : std::vector<FrameRange>{FrameRange(
                  vertex->safe_interval.first, vertex->safe_interval.second)};
        const auto intervals = as_frame_ranges(current_safe_interval_query(*vertex));
        const auto transition = classify_interval_set_transition(
            previous_intervals, intervals);
        switch (transition)
        {
        case IntervalSetTransition::Unchanged:
            ++outcome.report.intervals_unchanged;
            break;
        case IntervalSetTransition::Shrunk:
            ++outcome.report.intervals_shrunk;
            break;
        case IntervalSetTransition::Expanded:
            ++outcome.report.intervals_expanded;
            break;
        case IntervalSetTransition::Split:
            ++outcome.report.intervals_split;
            break;
        case IntervalSetTransition::Shifted:
            ++outcome.report.intervals_shifted;
            break;
        case IntervalSetTransition::Merged:
            ++outcome.report.intervals_merged;
            break;
        case IntervalSetTransition::Deleted:
            ++outcome.report.intervals_deleted;
            break;
        }
        const auto matching_interval = std::find_if(
            intervals.begin(),
            intervals.end(),
            [&](const auto &interval) {
                return interval.first <= vertex->arrival_time &&
                       vertex->arrival_time <= interval.last;
            });
        if (matching_interval == intervals.end())
        {
            invalid_roots.insert(vertex_id);
            continue;
        }
        tree.update_vertex_interval(
            vertex,
            {matching_interval->first, matching_interval->last},
            prediction_version);
    }

    for (const std::size_t edge_id : affected.edge_ids)
    {
        if (edge_id >= tree.array_of_vertices.size())
        {
            continue;
        }
        Vertex *child = tree.array_of_vertices[edge_id];
        if (!child->active || child->parent == nullptr || !child->parent->active)
        {
            continue;
        }
        ++outcome.report.revalidated_edges;
        if (!edge_validation(*child->parent, *child))
        {
            invalid_roots.insert(child->vertex_id);
        }
    }

    std::vector<std::size_t> roots(invalid_roots.begin(), invalid_roots.end());
    std::sort(roots.begin(), roots.end());
    for (const std::size_t root_id : roots)
    {
        Vertex *root = tree.array_of_vertices[root_id];
        if (!root->active || has_selected_ancestor(root, invalid_roots))
        {
            continue;
        }
        auto deactivated = tree.deactivate_subtree(root, prediction_version);
        outcome.invalidated_vertices.insert(
            outcome.invalidated_vertices.end(), deactivated.begin(), deactivated.end());
    }

    for (Vertex *vertex : tree.array_of_vertices)
    {
        if (vertex->active)
        {
            vertex->prediction_version = prediction_version;
        }
    }

    outcome.report.invalidated_vertices = outcome.invalidated_vertices.size();
    outcome.report.reuse_identity_initialized = true;
    for (const Vertex *vertex : tree.array_of_vertices)
    {
        if (vertex->active && active_vertex_ids_before.count(vertex->vertex_id) != 0)
        {
            outcome.report.reused_vertex_ids.insert(vertex->vertex_id);
        }
    }
    outcome.report.reused_vertices = outcome.report.reused_vertex_ids.size();
    outcome.report.invariant_error = describe_tree_invariant_violation(tree);
    outcome.report.invariants_hold = outcome.report.invariant_error.empty();
    return outcome;
}

bool check_tree_invariants(const Tree &tree)
{
    return describe_tree_invariant_violation(tree).empty();
}

std::string describe_tree_invariant_violation(const Tree &tree)
{
    std::size_t counted_active = 0;
    for (const Vertex *vertex : tree.array_of_vertices)
    {
        if (!vertex->active)
        {
            continue;
        }
        ++counted_active;
        if (vertex->arrival_time < vertex->safe_interval.first || vertex->arrival_time > vertex->safe_interval.second)
        {
            std::ostringstream message;
            message << "vertex " << vertex->vertex_id << " label " << vertex->arrival_time
                    << " is outside interval [" << vertex->safe_interval.first << ','
                    << vertex->safe_interval.second << ']';
            return message.str();
        }
        if (vertex->parent != nullptr)
        {
            if (!vertex->parent->active || vertex->parent->tree_id != vertex->tree_id)
            {
                return "vertex " + std::to_string(vertex->vertex_id) + " has an inactive or cross-tree parent";
            }
            if (std::find(vertex->parent->children.begin(), vertex->parent->children.end(), vertex) == vertex->parent->children.end())
            {
                return "vertex " + std::to_string(vertex->vertex_id) + " is absent from its parent's children";
            }
        }
    }
    if (counted_active != tree.active_vertex_count())
    {
        return "active vertex count mismatch: scanned=" + std::to_string(counted_active) +
               " tracked=" + std::to_string(tree.active_vertex_count());
    }
    return {};
}

} // namespace MDP::MSIRRT
