#include "PlannerConnect.hpp"
#include "config_read_writer/config_read.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: test_rolling_reroot <scene.json>\n";
        return 2;
    }

    MDP::ConfigReader reader(argv[1]);
    const auto initial = reader.get_scene_task();
    MDP::MSIRRT::PlannerConnect planner(initial, 42);
    const bool initially_solved = planner.solve();
    if (!initially_solved)
    {
        std::cerr << "initial planning failed\n";
        return 1;
    }
    const auto initial_path = planner.get_final_path();
    if (initial_path.size() < 2)
    {
        std::cerr << "initial path is too short for reroot testing\n";
        return 1;
    }
    const int execution_frame = std::max(
        1,
        static_cast<int>(std::floor(initial_path.back()->arrival_time / 2.0)));
    assert(execution_frame < initial_path.back()->arrival_time);

    const auto advance = planner.advance_start_to_frame(execution_frame);
    assert(advance.advanced);
    assert(!advance.goal_already_reached);
    assert(advance.execution_frame == execution_frame);
    assert(advance.configuration.size() == initial.start_configuration.size());
    assert(advance.start_tree.reused_vertices > 0);
    assert(advance.start_tree.invariants_hold);
    assert(planner.get_planning_start_frame() == execution_frame);

    const auto advanced_path = planner.get_final_path();
    assert(!advanced_path.empty());
    assert(std::abs(advanced_path.front()->arrival_time - execution_frame) < 1e-9);
    for (std::size_t joint = 0; joint < advance.configuration.size(); ++joint)
    {
        assert(std::abs(advanced_path.front()->coords[joint] - advance.configuration[joint]) < 1e-9);
    }

    auto next_prediction = initial;
    next_prediction.prediction_issue_frame = static_cast<unsigned int>(execution_frame);
    next_prediction.start_configuration = advance.configuration;
    const auto repair = planner.update_prediction(next_prediction);
    assert(repair.invariants_hold);
    assert(planner.get_planning_start_frame() == execution_frame);
    const bool solved_after_update = planner.solve();
    if (!solved_after_update)
    {
        std::cerr << "planning failed after reroot and prediction update\n";
        return 1;
    }
    assert(!planner.get_final_path().empty());

    auto already_at_goal = initial;
    already_at_goal.start_configuration = already_at_goal.end_configuration;
    MDP::MSIRRT::PlannerConnect goal_planner(already_at_goal, 42);
    const bool zero_length_solved = goal_planner.solve();
    if (!zero_length_solved || goal_planner.get_final_path().size() != 1)
    {
        std::cerr << "start-at-goal termination failed\n";
        return 1;
    }

    std::cout << "rolling reroot integration test passed\n";
    return 0;
}
