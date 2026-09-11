#pragma once

#include "RepairTypes.hpp"
#include "Tree.hpp"

#include <functional>
#include <utility>
#include <vector>

namespace MDP::MSIRRT
{

using SafeIntervalQuery = std::function<std::vector<std::pair<int, int>>(const Vertex &)>;
using EdgeValidation = std::function<bool(const Vertex &parent, const Vertex &child)>;

struct TreeRepairOutcome
{
    RepairReport report;
    std::vector<Vertex *> invalidated_vertices;
};

TreeRepairOutcome repair_tree(
    Tree &tree,
    const std::vector<FrameRange> &changed_windows,
    std::size_t prediction_version,
    const SafeIntervalQuery &previous_safe_interval_query,
    const SafeIntervalQuery &current_safe_interval_query,
    const EdgeValidation &edge_validation);

bool check_tree_invariants(const Tree &tree);
std::string describe_tree_invariant_violation(const Tree &tree);

} // namespace MDP::MSIRRT
