
#include <vector>
#include <cstdlib>

#include "PlannerConnect.hpp"
#include "config_read_writer/config_read.hpp"
#include "config_read_writer/SphereObstacleJsonInfo.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace
{

bool environment_flag(const char *name, bool fallback)
{
    const char *raw = std::getenv(name);
    return raw == nullptr ? fallback : std::atoi(raw) != 0;
}

} // namespace

MDP::MSIRRT::PlannerConnect::PlannerConnect(MDP::ConfigReader::SceneTask scene_task_, int random_seed) : scene_task(std::move(scene_task_)),
                                                                                                         collision_manager(std::make_unique<MDP::CollisionManager>(scene_task)), probability_gen(0.0, 1.0), conditional_probability_gen(0.0, 1.0),
                                                                                                         gen(random_seed), conditional_gen(static_cast<std::mt19937::result_type>(random_seed) ^ 0x9e3779b9U), rd()
{
    this->horizon_bounded_safe_interval_queries_enabled = environment_flag(
        "MSIRRT_HORIZON_BOUNDED_SAFE_INTERVALS", false);
    this->dof = scene_task.start_configuration.size();
    this->planning_start_frame = static_cast<int>(scene_task.prediction_issue_frame);
    this->planning_horizon_frame = static_cast<int>(std::min(
        scene_task.reliable_until_frame,
        scene_task.frame_count - 1));
    this->start_tree = new MDP::MSIRRT::Tree("start_tree", 0, this->dof);
    this->goal_tree = new MDP::MSIRRT::Tree("goal_tree", 1, this->dof);
    this->orphan_tree = new MDP::MSIRRT::Tree("orphan_tree", -1, this->dof);
    this->start_tree->array_of_vertices.reserve(1000000);  // TODO: decide to leave it or remove it
    this->goal_tree->array_of_vertices.reserve(1000000);   // TODO: decide to leave it or remove it
    this->orphan_tree->array_of_vertices.reserve(1000000); // TODO: decide to leave it or remove it

    this->robot_limits = this->collision_manager->get_planned_robot_limits();

    assert(is_coords_in_limits(MDP::MSIRRT::Vertex::VertexCoordType(scene_task.start_configuration.data()))); // check start conf bound;

    assert(is_coords_in_limits(MDP::MSIRRT::Vertex::VertexCoordType(scene_task.end_configuration.data()))); // check end conf bounds;

    // std::cout<<"--------------------------------------------"<<std::endl;
    // std::cout<<"Get staret tree safe intervals!!!!"<<std::endl;
    std::vector<std::pair<int, int>> start_safe_intervals = this->get_planning_safe_intervals(scene_task.start_configuration); // get start cond save intervals;

    const auto start_interval = std::find_if(
        start_safe_intervals.begin(),
        start_safe_intervals.end(),
        [&](const auto &interval) {
            return interval.first <= this->planning_start_frame &&
                   this->planning_start_frame <= interval.second;
        });
    if (start_interval == start_safe_intervals.end())
    {
        throw std::runtime_error("start configuration is not safe at prediction issue frame");
    }

    this->start_tree->add_vertex(
        scene_task.start_configuration,
        *start_interval,
        nullptr,
        -1,
        this->planning_start_frame);
    
    // std::cout<<"start safe intervals:"<<std::endl;
    // for (int another_safe_interval_id = 1; another_safe_interval_id < start_safe_intervals.size(); another_safe_interval_id++)
    // {
    //     // std::cout<< start_safe_intervals[another_safe_interval_id].first<<" "<<start_safe_intervals[another_safe_interval_id].second<<std::endl;
    //     this->orphan_tree->add_vertex(scene_task.start_configuration, start_safe_intervals[another_safe_interval_id], nullptr, -1, -1);
    // }

    this->root_node = this->start_tree->array_of_vertices[0];
    this->root_node->arrival_time = this->planning_start_frame;

    this->goal_coords = MDP::MSIRRT::Vertex::VertexCoordType(scene_task.end_configuration.data());
    this->goal_safe_intervals = this->get_planning_safe_intervals(scene_task.end_configuration);

    assert(this->goal_safe_intervals.size()>0);
    if (this->goal_safe_intervals.size()==0){
        std::cout<<"this->goal_safe_intervals.size()==0" <<std::endl;
        std::exit(1);
    }
    // std::cout<<"goal safe intervals:"<<std::endl;
    for (const std::pair<int,int>& safe_int:this->goal_safe_intervals){
    //     std::cout<< safe_int.first<<" "<<safe_int.second<<std::endl;
        this->goal_tree->add_vertex(this->goal_coords, safe_int, nullptr, -1, safe_int.second);
    
    }
    //// this->goal_tree->add_vertex(this->goal_coords, this->goal_safe_intervals.back(), nullptr, -1, this->goal_safe_intervals.back().second);
    
    // сheck if the latest safe interval at goal is safe
    // assert(this->goal_safe_intervals.back().second == scene_task.frame_count - 1);
    this->current_tree = this->goal_tree;
    this->other_tree = this->start_tree;
    this->goal_reached = this->is_goal(this->root_node->coords);
    if (this->goal_reached)
    {
        this->finish_node = this->root_node;
    }
    this->max_planning_time = 20;      // TODO: remove hardcode
    this->stop_when_path_found = true; // TODO: remove hardcode
    this->planner_range = 1;           // TODO: remove hardcode
    this->vmax = MDP::MSIRRT::DEFAULT_MAXIMUM_JOINT_SPACE_SPEED;

    // Allow overrides from environment for parameter sweeps without recompiles.
    // MSIRRT_PLANNER_RANGE: step length; MSIRRT_RADIUS_FACTOR: set_parent
    // search radius squared = factor * range^2.
    if (const char *env = std::getenv("MSIRRT_PLANNER_RANGE")) {
        double v = std::atof(env);
        if (v > 0) this->planner_range = v;
    }
    // Default radius_factor = 4.0 (i.e. set_parent searches a ball of radius
    // 2 * planner_range). Picked via parameter sweep over 1-200 obstacles —
    // it consistently beat the legacy 9.0 and 25.0 settings on small-to-medium
    // scenes by 1.5-2.5x while preserving 100% success on dense scenes.
    if (const char *env = std::getenv("MSIRRT_RADIUS_FACTOR")) {
        double v = std::atof(env);
        if (v > 0) this->radius_factor = v;
    }
    // MSIRRT_MAX_DEP_TIME: hysteresis frames around the static-obstacle "jump"
    // trigger in set_parent. 0 = jump aggressively on first collision (MSIRRT_jump
    // baseline). Higher values delay the jump and try more departure_time values
    // first — denser scenes may prefer this to avoid noisy get_safe_intervals
    // calls. Negative values disabled.
    if (const char *env = std::getenv("MSIRRT_MAX_DEP_TIME")) {
        int v = std::atoi(env);
        if (v >= 0) this->max_dep_time_for_safe_int_check = v;
    }
    if (const char *env = std::getenv("MSIRRT_MAX_PLANNING_TIME")) {
        double value = std::atof(env);
        if (value > 0.0) this->max_planning_time = value;
    }
    if (const char *env = std::getenv("MSIRRT_CONDITIONAL_SAMPLING")) {
        this->conditional_sampling_enabled = std::atoi(env) != 0;
    }
    if (const char *env = std::getenv("MSIRRT_CONDITIONAL_EXPLORATION_RATE")) {
        const double value = std::atof(env);
        if (value >= 0.0 && value <= 1.0) this->conditional_exploration_rate = value;
    }
    if (const char *env = std::getenv("MSIRRT_REJECTION_FEEDBACK")) {
        this->rejection_feedback_enabled = std::atoi(env) != 0;
    }
    if (const char *env = std::getenv("MSIRRT_REUSE_STORED_PREVIOUS_INTERVALS")) {
        this->reuse_stored_previous_intervals = std::atoi(env) != 0;
    }
    if (const char *env = std::getenv("MSIRRT_COOPERATIVE_DEADLINE_CHECKS")) {
        this->cooperative_deadline_checks_enabled = std::atoi(env) != 0;
    }
    if (const char *env = std::getenv("MSIRRT_REACHABLE_ELLIPSOID_SAMPLING")) {
        this->reachable_ellipsoid_sampling_enabled = std::atoi(env) != 0;
    }
    if (const char *env = std::getenv("MSIRRT_REJECTION_WINDOW")) {
        const int value = std::atoi(env);
        if (value > 0) this->conditional_rejection_window = static_cast<std::size_t>(value);
    }
    if (const char *env = std::getenv("MSIRRT_REJECTION_THRESHOLD")) {
        const double value = std::atof(env);
        if (value >= 0.0 && value <= 1.0) this->conditional_rejection_threshold = value;
    }
    if (const char *env = std::getenv("MSIRRT_REJECTION_EXPLORATION_BOOST")) {
        const double value = std::atof(env);
        if (value >= 0.0 && value <= 1.0) this->conditional_exploration_boost = value;
    }
    if (const char *env = std::getenv("MSIRRT_ADAPTIVE_REPAIR_POLICY")) {
        if (std::string(env) == "legacy") this->rejection_feedback_enabled = false;
    }
    this->conditional_feedback = std::make_unique<MDP::MSIRRT::ConditionalExplorationFeedback>(
        this->conditional_exploration_rate,
        this->conditional_exploration_boost,
        this->conditional_rejection_window,
        this->conditional_rejection_threshold,
        this->rejection_feedback_enabled);
}
// destructor
MDP::MSIRRT::PlannerConnect::~PlannerConnect()
{
    delete this->start_tree;
    delete this->goal_tree;
    delete this->orphan_tree;
}

bool MDP::MSIRRT::PlannerConnect::solve()
{
    
    this->solver_start_time = std::chrono::steady_clock::now();
    struct SearchBudgetScope
    {
        bool &active;
        explicit SearchBudgetScope(bool &value) : active(value) { active = true; }
        ~SearchBudgetScope() { active = false; }
    } search_budget_scope(this->search_budget_active);
    int iter = 0;

    if (this->root_node != nullptr && this->root_node->active && this->is_goal(this->root_node->coords))
    {
        this->goal_reached = true;
        this->finish_node = this->root_node;
        return true;
    }

    if (this->root_node == nullptr || !this->root_node->active ||
        this->start_tree->active_vertex_count() == 0 || this->goal_tree->active_vertex_count() == 0)
    {
        return false;
    }
    
    if(this->root_node->safe_interval.second < this->planning_horizon_frame/5){
        this->warmup_start_tree();
    }

    while (this->check_planner_termination_condition() && !this->goal_reached)
    {
        // std::cout << "iteration #" << iter++ << std::endl;
        MDP::MSIRRT::Vertex::VertexCoordType coord_rand;
        coord_rand = get_random_configuration();
        ++this->proposed_samples;

        bool conditional_outcome_recorded = false;
        const auto record_conditional_outcome = [&](bool rejected) {
            if (!this->conditional_sampling_enabled || conditional_outcome_recorded)
            {
                return;
            }
            this->conditional_feedback->observe(rejected);
            conditional_outcome_recorded = true;
        };

        if (this->conditional_sampling_enabled && !this->is_sample_kinematically_reachable(coord_rand))
        {
            const double exploration_rate = this->conditional_feedback->effective_rate();
            if (this->conditional_feedback->active())
            {
                ++this->conditional_feedback_boosted_samples;
            }
            const bool bypass =
                this->conditional_probability_gen(this->conditional_gen) < exploration_rate;
            record_conditional_outcome(true);
            if (bypass)
            {
                ++this->conditionally_bypassed_samples;
            }
            else
            {
                ++this->conditionally_rejected_samples;
                ++this->conditionally_rejected_kinematic_samples;
                this->swap_trees();
                continue;
            }
        }

        bool is_ok = this->extend(coord_rand);
        if (this->cooperative_search_budget_exhausted())
        {
            break;
        }
        if (!is_ok)
        {
            record_conditional_outcome(false);
            continue;
        }

        std::vector<double> robot_angles(coord_rand.data(), coord_rand.data() + coord_rand.rows() * coord_rand.cols());

        std::vector<std::pair<int, int>> safe_intervals_of_coord_rand;

        safe_intervals_of_coord_rand = this->get_planning_safe_intervals(robot_angles);
        if (this->cooperative_search_budget_exhausted())
        {
            break;
        }
        ++this->evaluated_samples;
        if (this->conditional_sampling_enabled && !this->is_sample_temporally_reachable(coord_rand, safe_intervals_of_coord_rand))
        {
            const double exploration_rate = this->conditional_feedback->effective_rate();
            if (this->conditional_feedback->active())
            {
                ++this->conditional_feedback_boosted_samples;
            }
            const bool bypass =
                this->conditional_probability_gen(this->conditional_gen) < exploration_rate;
            record_conditional_outcome(true);
            if (bypass)
            {
                ++this->conditionally_bypassed_samples;
            }
            else
            {
                ++this->conditionally_rejected_samples;
                ++this->conditionally_rejected_temporal_samples;
                this->swap_trees();
                continue;
            }
        }
        record_conditional_outcome(false);

        std::vector<MDP::MSIRRT::Vertex *> new_nodes = this->grow_tree(coord_rand, safe_intervals_of_coord_rand);
        if (this->cooperative_search_budget_exhausted())
        {
            break;
        }

        // if (new_nodes.size() == 0 && this->was_static_obstacle)
        // {
        //     coord_rand = this->last_valid_coord;
        //     safe_intervals_of_coord_rand = this->collision_manager.get_safe_intervals(std::vector<double>(coord_rand.data(), coord_rand.data() + coord_rand.rows() * coord_rand.cols()));
        //     new_nodes = this->grow_tree(coord_rand, safe_intervals_of_coord_rand);
        // }   
        // assert(this->was_static_obstacle);
        if(new_nodes.size() == 0){
            this->swap_trees();
            continue;
        }
        bool connected = connect_trees(coord_rand, safe_intervals_of_coord_rand, new_nodes);
        this->swap_trees();
    }
    return this->goal_reached;
}

bool MDP::MSIRRT::PlannerConnect::has_valid_solution() const
{
    return this->goal_reached && !this->get_final_path().empty();
}

void MDP::MSIRRT::PlannerConnect::warmup_start_tree()
{
    // std::cout<<"warmup_start_tree"<<std::endl;
    this->current_tree = this->start_tree;
    this->other_tree = this->goal_tree;
    double old_planner_range = this->planner_range;
    this->planner_range = 1.5*this->root_node->safe_interval.second/this->scene_task.fps * this->vmax;
    for(int i = 0; i < 25; i++){
        if (this->cooperative_search_budget_exhausted())
        {
            break;
        }
        MDP::MSIRRT::Vertex::VertexCoordType coord_rand;
        coord_rand = get_random_configuration();

        bool is_ok = this->extend(coord_rand);
        if (!is_ok)
        {
            continue;
        }

        std::vector<double> robot_angles(coord_rand.data(), coord_rand.data() + coord_rand.rows() * coord_rand.cols());

        std::vector<std::pair<int, int>> safe_intervals_of_coord_rand;

        safe_intervals_of_coord_rand = this->get_planning_safe_intervals(robot_angles);

        std::vector<MDP::MSIRRT::Vertex *> new_nodes = this->grow_tree(coord_rand, safe_intervals_of_coord_rand);

        if (new_nodes.size() == 0 && this->was_static_obstacle)
        {
            coord_rand = this->last_valid_coord;
            safe_intervals_of_coord_rand = this->get_planning_safe_intervals(std::vector<double>(coord_rand.data(), coord_rand.data() + coord_rand.rows() * coord_rand.cols()));
            new_nodes = this->grow_tree(coord_rand, safe_intervals_of_coord_rand);
        }   
        // if (new_nodes.size() != 0){
        //     std::cout<<"warmup_start_tree: new_nodes.size() != 0"<<std::endl;
        // }
    }

    this->planner_range = old_planner_range;
}


void MDP::MSIRRT::PlannerConnect::swap_trees()
{
    std::swap(this->current_tree, this->other_tree);
}

bool MDP::MSIRRT::PlannerConnect::connect_trees(MDP::MSIRRT::Vertex::VertexCoordType &coord_rand, std::vector<std::pair<int, int>> &safe_intervals_of_coord_rand, std::vector<MDP::MSIRRT::Vertex *> another_tree_new_nodes)
{
    const MDP::MSIRRT::Vertex::VertexCoordType original_coord_rand = coord_rand;

    while (this->check_planner_termination_condition())
    { 
        coord_rand = original_coord_rand;
        this->swap_trees();
        bool is_ok = this->extend(coord_rand);
        this->swap_trees();
        if (!is_ok)
        {
            break;
        }

        std::vector<double> robot_angles(coord_rand.data(), coord_rand.data() + coord_rand.rows() * coord_rand.cols());

        std::vector<std::pair<int, int>> safe_intervals_of_coord_rand;

        safe_intervals_of_coord_rand = this->get_planning_safe_intervals(robot_angles);
        this->swap_trees(); 
        std::vector<MDP::MSIRRT::Vertex *> new_nodes = this->grow_tree(coord_rand, safe_intervals_of_coord_rand);
        // if (new_nodes.size() == 0 && this->was_static_obstacle)
        // {
        //     coord_rand = this->last_valid_coord;
        //     safe_intervals_of_coord_rand = this->collision_manager.get_safe_intervals(std::vector<double>(coord_rand.data(), coord_rand.data() + coord_rand.rows() * coord_rand.cols()));
        //     new_nodes = this->grow_tree(coord_rand, safe_intervals_of_coord_rand);
        //     // assert(this->was_static_obstacle);
        //     break; // because we hit static obstacle

        // }   
        this->swap_trees();
        if (new_nodes.size() == 0)
        {
            break;
        }

        if (original_coord_rand == coord_rand) // if we reached original nodes of first tree
        {
            for (MDP::MSIRRT::Vertex *node : new_nodes) // for new node in second tree
            {
                for (MDP::MSIRRT::Vertex *another_tree_node : another_tree_new_nodes) // for new node in first tree
                {
                    // assert(node->coords == another_tree_node->coords);
                    if (node->coords == another_tree_node->coords)
                    {
                        if (node->safe_interval == another_tree_node->safe_interval)
                        {
                            if (node->tree_id == 1) // TODO: change tree_id to enums. 1==goal_Tree
                            {
                                if (node->arrival_time >= another_tree_node->arrival_time)
                                {
                                    this->goal_reached = true;
                                    this->goal_nodes = std::pair<MDP::MSIRRT::Vertex *, MDP::MSIRRT::Vertex *>(another_tree_node, node);
                                    // std::cout << "goal reached!" << std::endl;
                                    //
                                    this->prune_goal_tree();
                                }
                            }
                            else
                            {
                                if (node->arrival_time <= another_tree_node->arrival_time)
                                {
                                    this->goal_reached = true;
                                    this->goal_nodes = std::pair<MDP::MSIRRT::Vertex *, MDP::MSIRRT::Vertex *>(node, another_tree_node);
                                    // std::cout << "goal reached!" << std::endl;
                                    this->prune_goal_tree();
                                }
                            }
                            return true;
                        }
                    }
                }
                if (this->goal_reached)
                {
                    break;
                }
            }
            break;
        }
        coord_rand = original_coord_rand;
    }

    return false;
}

std::vector<MDP::MSIRRT::Vertex *> MDP::MSIRRT::PlannerConnect::grow_tree(MDP::MSIRRT::Vertex::VertexCoordType &coord_rand, std::vector<std::pair<int, int>> &safe_intervals_of_coord_rand)
{
    std::vector<MDP::MSIRRT::Vertex *> result;
    std::vector<MDP::MSIRRT::Vertex *> array_of_new_nodes = this->set_parent(coord_rand, safe_intervals_of_coord_rand);

    if (array_of_new_nodes.size() == 0)
    {
        // std::cout << "no_new_node" << std::endl;
        return array_of_new_nodes;
    }
    // result.reserve(result.size() + std::distance(array_of_new_nodes.begin(), array_of_new_nodes.end()));
    // result.insert(result.end(), array_of_new_nodes.begin(), array_of_new_nodes.end());

    // std::cout << "find_new_node!" << std::endl;
    // std::cout << "tree size:" << this->start_tree->array_of_vertices.size() << " " << this->goal_tree->array_of_vertices.size() << "  " << this->orphan_tree->array_of_vertices.size() << std::endl;

    return array_of_new_nodes;
}

std::vector<MDP::MSIRRT::Vertex *> MDP::MSIRRT::PlannerConnect::get_final_path() const
{
    std::vector<MDP::MSIRRT::Vertex *> result;
    if (!this->goal_reached)
    {
        return result;
    }
    MDP::MSIRRT::Vertex *current = this->finish_node;

    while (current)
    {
        result.push_back(current);
        current = current->parent;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<double> MDP::MSIRRT::PlannerConnect::configuration_at_frame(double frame) const
{
    const auto path = this->get_final_path();
    if (path.empty())
    {
        throw std::runtime_error("cannot sample a configuration before a solution exists");
    }
    MDP::MSIRRT::Vertex::VertexCoordType configuration = path.front()->coords;
    if (frame <= path.front()->arrival_time)
    {
        return std::vector<double>(configuration.data(), configuration.data() + configuration.size());
    }
    for (std::size_t index = 1; index < path.size(); ++index)
    {
        const auto *parent = path[index - 1];
        const auto *child = path[index];
        if (frame <= child->departure_from_parent_time)
        {
            configuration = parent->coords;
            return std::vector<double>(configuration.data(), configuration.data() + configuration.size());
        }
        if (frame <= child->arrival_time)
        {
            const double duration = child->arrival_time - child->departure_from_parent_time;
            const double alpha = duration > 0.0
                ? std::clamp((frame - child->departure_from_parent_time) / duration, 0.0, 1.0)
                : 1.0;
            configuration = parent->coords + (child->coords - parent->coords) * alpha;
            return std::vector<double>(configuration.data(), configuration.data() + configuration.size());
        }
        configuration = child->coords;
    }
    return std::vector<double>(configuration.data(), configuration.data() + configuration.size());
}

MDP::MSIRRT::ExecutionAdvanceReport MDP::MSIRRT::PlannerConnect::advance_start_to_frame(int frame)
{
    MDP::MSIRRT::ExecutionAdvanceReport report;
    report.execution_frame = frame;
    const auto path = this->get_final_path();
    if (path.empty())
    {
        throw std::runtime_error("cannot advance execution before a solution exists");
    }
    if (frame < this->planning_start_frame)
    {
        throw std::invalid_argument("execution frame cannot move backwards");
    }
    report.configuration = this->configuration_at_frame(frame);
    if (frame > path.back()->arrival_time)
    {
        report.goal_already_reached = true;
        return report;
    }
    if (frame > this->planning_horizon_frame)
    {
        throw std::invalid_argument("execution frame exceeds the currently validated planning horizon");
    }

    MDP::MSIRRT::Vertex *anchor = nullptr;
    MDP::MSIRRT::Vertex *edge_child = nullptr;
    constexpr double tolerance = 1e-9;
    if (std::abs(frame - path.front()->arrival_time) <= tolerance)
    {
        anchor = path.front();
    }
    for (std::size_t index = 1; index < path.size() && anchor == nullptr && edge_child == nullptr; ++index)
    {
        MDP::MSIRRT::Vertex *parent = path[index - 1];
        MDP::MSIRRT::Vertex *child = path[index];
        if (frame <= child->departure_from_parent_time + tolerance)
        {
            anchor = parent;
        }
        else if (frame < child->arrival_time - tolerance)
        {
            edge_child = child;
        }
        else if (std::abs(frame - child->arrival_time) <= tolerance)
        {
            anchor = child;
        }
    }
    if (anchor == nullptr && edge_child == nullptr)
    {
        anchor = path.back();
    }

    const auto raw_intervals = this->horizon_bounded_safe_interval_queries_enabled
        ? this->collision_manager->get_safe_intervals(
              report.configuration, this->planning_horizon_frame)
        : this->collision_manager->get_safe_intervals(report.configuration);
    const auto interval = std::find_if(raw_intervals.begin(), raw_intervals.end(), [&](const auto &candidate) {
        return candidate.first <= frame && frame <= candidate.second;
    });
    if (interval == raw_intervals.end())
    {
        throw std::runtime_error("executed configuration is not safe at the requested frame");
    }
    const std::pair<int, int> root_interval{
        frame,
        std::min(interval->second, this->planning_horizon_frame)};
    if (anchor != nullptr)
    {
        report.start_tree = this->start_tree->reroot_at_vertex(
            anchor,
            root_interval,
            frame,
            this->prediction_version);
    }
    else
    {
        report.start_tree = this->start_tree->reroot_on_edge(
            edge_child,
            MDP::MSIRRT::Vertex::VertexCoordType(report.configuration.data()),
            root_interval,
            frame,
            this->prediction_version);
    }

    this->root_node = report.start_tree.root;
    this->planning_start_frame = frame;
    this->scene_task.prediction_issue_frame = static_cast<unsigned int>(frame);
    this->scene_task.start_configuration = report.configuration;
    this->executed_prefix = true;
    report.advanced = true;
    return report;
}

bool MDP::MSIRRT::PlannerConnect::check_planner_termination_condition() const
{
    const double elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - this->solver_start_time).count();
    if (this->stop_when_path_found)
    {
        return !this->goal_reached && elapsed_seconds < this->max_planning_time;
    }
    return elapsed_seconds < this->max_planning_time;
}

bool MDP::MSIRRT::PlannerConnect::cooperative_search_budget_exhausted()
{
    if (!this->cooperative_deadline_checks_enabled || !this->search_budget_active ||
        this->check_planner_termination_condition())
    {
        return false;
    }
    ++this->cooperative_deadline_check_triggers;
    return true;
}

void MDP::MSIRRT::PlannerConnect::set_max_planning_time(double seconds)
{
    if (!std::isfinite(seconds) || seconds <= 0.0)
    {
        throw std::invalid_argument("planning time budget must be finite and positive");
    }
    this->max_planning_time = static_cast<float>(seconds);
}

bool MDP::MSIRRT::PlannerConnect::get_cooperative_deadline_checks_enabled() const
{
    return this->cooperative_deadline_checks_enabled;
}

std::size_t MDP::MSIRRT::PlannerConnect::get_cooperative_deadline_check_triggers() const
{
    return this->cooperative_deadline_check_triggers;
}

bool MDP::MSIRRT::PlannerConnect::get_reachable_ellipsoid_sampling_enabled() const
{
    return this->reachable_ellipsoid_sampling_enabled;
}

std::size_t MDP::MSIRRT::PlannerConnect::get_reachable_ellipsoid_samples() const
{
    return this->reachable_ellipsoid_samples;
}

bool MDP::MSIRRT::PlannerConnect::is_coords_in_limits(const MDP::MSIRRT::Vertex &q) const
{
    return this->is_coords_in_limits(q.coords);
}
bool MDP::MSIRRT::PlannerConnect::is_coords_in_limits(const MDP::MSIRRT::Vertex::VertexCoordType &coords) const
{
    // assert(this->robot_limits.size() == coords.size());
    for (int joint_ind = 0; joint_ind < this->robot_limits.size(); joint_ind++)
    {
        if ((coords[joint_ind] < robot_limits[joint_ind].first) | (coords[joint_ind] > robot_limits[joint_ind].second))
        {
            return false;
        }
    }
    return true;
}

double MDP::MSIRRT::PlannerConnect::get_random_between_0_1()
{
    return this->probability_gen(this->gen);
}
MDP::MSIRRT::Vertex::VertexCoordType MDP::MSIRRT::PlannerConnect::get_random_configuration()
{
    if (this->conditional_sampling_enabled && this->reachable_ellipsoid_sampling_enabled)
    {
        return this->get_reachable_ellipsoid_configuration();
    }
    return this->get_uniform_random_configuration();
}

MDP::MSIRRT::Vertex::VertexCoordType MDP::MSIRRT::PlannerConnect::get_uniform_random_configuration()
{
    MDP::MSIRRT::Vertex::VertexCoordType result;

    for (int joint_ind = 0; joint_ind < this->dof; joint_ind++)
    {
        result[joint_ind] = this->get_random_between_0_1() * (this->robot_limits[joint_ind].second - robot_limits[joint_ind].first) + robot_limits[joint_ind].first;
    }
    return result;
}

MDP::MSIRRT::Vertex::VertexCoordType MDP::MSIRRT::PlannerConnect::get_reachable_ellipsoid_configuration()
{
    const MDP::MSIRRT::Vertex::VertexCoordType start(
        this->scene_task.start_configuration.data());
    const MDP::MSIRRT::Vertex::VertexCoordType center =
        0.5 * (start + this->goal_coords);
    const MDP::MSIRRT::Vertex::VertexCoordType focal_vector =
        this->goal_coords - start;
    const double focal_distance = focal_vector.norm();
    const double available_distance =
        static_cast<double>(this->planning_horizon_frame - this->planning_start_frame) *
        this->vmax / static_cast<double>(this->scene_task.fps);
    if (!std::isfinite(available_distance) ||
        available_distance + 1e-12 < focal_distance)
    {
        return this->get_uniform_random_configuration();
    }

    std::normal_distribution<double> standard_normal(0.0, 1.0);
    MDP::MSIRRT::Vertex::VertexCoordType local;
    double norm = 0.0;
    do
    {
        for (int joint_ind = 0; joint_ind < this->dof; ++joint_ind)
        {
            local[joint_ind] = standard_normal(this->gen);
        }
        norm = local.norm();
    } while (norm <= std::numeric_limits<double>::epsilon());
    const double radius = std::pow(
        this->get_random_between_0_1(), 1.0 / static_cast<double>(this->dof));
    local *= radius / norm;

    const double major_radius = 0.5 * available_distance;
    const double focal_radius = 0.5 * focal_distance;
    const double minor_radius = std::sqrt(std::max(
        0.0, major_radius * major_radius - focal_radius * focal_radius));
    local[0] *= major_radius;
    for (int joint_ind = 1; joint_ind < this->dof; ++joint_ind)
    {
        local[joint_ind] *= minor_radius;
    }

    if (focal_distance > std::numeric_limits<double>::epsilon())
    {
        MDP::MSIRRT::Vertex::VertexCoordType reflection_vector =
            MDP::MSIRRT::Vertex::VertexCoordType::Unit(0) -
            focal_vector / focal_distance;
        const double reflection_norm_squared = reflection_vector.squaredNorm();
        if (reflection_norm_squared > std::numeric_limits<double>::epsilon())
        {
            local -= 2.0 * reflection_vector *
                (reflection_vector.dot(local) / reflection_norm_squared);
        }
    }

    MDP::MSIRRT::Vertex::VertexCoordType result = center + local;
    for (int joint_ind = 0; joint_ind < this->dof; ++joint_ind)
    {
        result[joint_ind] = std::clamp(
            result[joint_ind],
            static_cast<double>(this->robot_limits[joint_ind].first),
            static_cast<double>(this->robot_limits[joint_ind].second));
    }
    ++this->reachable_ellipsoid_samples;
    return result;
}

bool MDP::MSIRRT::PlannerConnect::extend(MDP::MSIRRT::Vertex::VertexCoordType &coords_of_new)
{
    MDP::MSIRRT::Vertex *q_nearest = this->get_nearest_node(coords_of_new); // find nearest neighbor
    if (q_nearest == nullptr)
    {
        return false;
    }

    // find new coords
    MDP::MSIRRT::Vertex::VertexCoordType delta_vector = coords_of_new - q_nearest->coords;
    double delta = delta_vector.norm();
    if (delta <= (1 / 100000000)) // zero division check
    {
        return false;
    }

    if (delta < this->planner_range)
    {
        return true;
    }

    coords_of_new = q_nearest->coords + delta_vector.normalized() * this->planner_range;

    return true;
}
MDP::MSIRRT::Vertex *MDP::MSIRRT::PlannerConnect::get_nearest_node(const MDP::MSIRRT::Vertex::VertexCoordType &coords)
{
    if (this->current_tree->active_vertex_count() == 0)
    {
        return nullptr;
    }
    const size_t num_results = 1;
    size_t ret_index;
    double out_dist_sqr;
    // nanoflann::SearchParams search_params;
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(&ret_index, &out_dist_sqr);
    this->current_tree->kd_tree.findNeighbors(resultSet, coords.data(), {0});
    return this->current_tree->array_of_vertices[ret_index];
}

std::vector<std::pair<MDP::MSIRRT::Vertex *, int>> MDP::MSIRRT::PlannerConnect::get_nearest_node_by_radius(const MDP::MSIRRT::Vertex::VertexCoordType &coords, double radius, MDP::MSIRRT::Tree *tree)
{
    std::vector<std::pair<MDP::MSIRRT::Vertex *, int>> result;

    // Unsorted radius search
    std::vector<nanoflann::ResultItem<size_t, double>> indices_dists;
    // search_params.sorted = false;
    nanoflann::RadiusResultSet<double, size_t> resultSet(radius, indices_dists);

    tree->kd_tree.findNeighbors(resultSet, coords.data(), {0});

    result.reserve(resultSet.m_indices_dists.size());
    for (auto node_ind_dist_pair : resultSet.m_indices_dists)
    {
        MDP::MSIRRT::Vertex *vertex = tree->array_of_vertices[node_ind_dist_pair.first];
        if (vertex->active)
        {
            result.emplace_back(vertex, node_ind_dist_pair.first);
        }
    }

    return result;
}

std::vector<MDP::MSIRRT::Vertex *> MDP::MSIRRT::PlannerConnect::set_parent(MDP::MSIRRT::Vertex::VertexCoordType &coord_rand, std::vector<std::pair<int, int>> &safe_intervals_of_coord_rand)
{
    if (this->current_tree == this->goal_tree &&
        (this->root_node == nullptr || !this->root_node->active))
    {
        return {};
    }

    this->was_static_obstacle = false;
    std::vector<std::pair<MDP::MSIRRT::Vertex *, int>> nearest_nodes = this->get_nearest_node_by_radius(coord_rand, this->radius_factor * this->planner_range * this->planner_range, (this->current_tree));

    std::vector<MDP::MSIRRT::Vertex *> added_vertices;
    // int safe_interval_ind = 0;
    
    
    if (this->current_tree == this->start_tree)
    {   
        // std::cout<<"set_parent start_tree"<<std::endl;
        double time_to_goal = (coord_rand - this->goal_coords).norm() * (double)this->scene_task.fps / this->vmax;
        // for (const std::pair<MDP::MSIRRT::Vertex *, int> &node : nearest_nodes){
        //     if (!node.first){
        //         std::cout<<"null pointer in sort!"<<std::endl;
        //         assert(false);
        //     }
        // }
        std::sort(nearest_nodes.begin(), nearest_nodes.end(), [&coord_rand,this](const std::pair<MDP::MSIRRT::Vertex *, int> &a, std::pair<MDP::MSIRRT::Vertex *, int> &b)
                  { 
                    // Add null pointer checks to prevent segfault
                    if (!a.first || !b.first) {
                        std::cout<<"null pointer in sort!"<<std::endl;

                        
                        return a.first != nullptr; // null pointers go to the end
                    }
                    return a.first->arrival_time+((coord_rand - a.first->coords).norm() * (double)this->scene_task.fps / this->vmax) < b.first->arrival_time+((coord_rand - b.first->coords).norm() * (double)this->scene_task.fps / this->vmax); 
                  });
                  // std::sort(nearest_nodes.begin(), nearest_nodes.end(), [](const std::pair<MDP::MSIRRT::Vertex *, int> &a, std::pair<MDP::MSIRRT::Vertex *, int> &b)
        //           { return a.first->arrival_time < b.first->arrival_time; });

        // for (std::pair<MDP::MSIRRT::Vertex *, int> candidate_node : nearest_nodes){
        //     std::cout<<candidate_node.first->arrival_time<<", ";
        // }
        // std::cout<<std::endl;
        for (std::pair<int, int> safe_int : safe_intervals_of_coord_rand) // For each interval
        {
            if (this->cooperative_search_budget_exhausted())
            {
                return added_vertices;
            }
            // std::cout<<"safe_int: "<<safe_int.first<<" "<<safe_int.second<<std::endl;
            // if we can't reach goal from that interval - skip
            if ((safe_int.first + time_to_goal) > this->planning_horizon_frame)
            {
                // safe_interval_ind++;
                continue;
            }
            bool found_parent = false;

            // for each parent
            for (std::pair<MDP::MSIRRT::Vertex *, int> candidate_node : nearest_nodes)
            {
                if (this->cooperative_search_budget_exhausted())
                {
                    return added_vertices;
                }

                double time_to_node = (coord_rand - candidate_node.first->coords).norm() * (double)this->scene_task.fps / this->vmax;

                int max_dep_time_for_safe_int_check = this->max_dep_time_for_safe_int_check;
                int col_coord_safe_int_first = std::max(candidate_node.first->arrival_time, (double)safe_int.first - time_to_node) + max_dep_time_for_safe_int_check;
                int col_coord_safe_int_second = col_coord_safe_int_first-1;

                // std::cout<<"candidate_node: "<<candidate_node.first->arrival_time<<" "<<candidate_node.first->safe_interval.first<<" "<<candidate_node.first->safe_interval.second<<std::endl;

                if (time_to_node<1){ // duplicate... this point was already added
                    found_parent = true;
                    break;
                }
                // candidate nodes are sorted by ascending arrival time. If arrival time > safe int bound + time to node -> break;
                if (candidate_node.first->arrival_time + time_to_node > safe_int.second)
                {
                    break;
                }

                // if intervals don't overlap - skip
                if ((candidate_node.first->safe_interval.second + time_to_node < safe_int.first))
                {
                    continue;
                }
                // assert(candidate_node.first);

                MDP::MSIRRT::Vertex::VertexCoordType start_coords = candidate_node.first->coords;

                // for each departure time
                for (double departure_time = std::max(candidate_node.first->arrival_time, (double)safe_int.first - time_to_node); departure_time <= std::min((double)candidate_node.first->safe_interval.second, (double)safe_int.second - time_to_node); departure_time += 1)
                {
                    if (this->cooperative_search_budget_exhausted())
                    {
                        return added_vertices;
                    }
                    // std::cout<<"departure_time: "<<departure_time<<std::endl;
                    // assert(!is_collision_motion(candidate_node.first->coords, candidate_node.first->coords, candidate_node.first->arrival_time, departure_time));
                    double arrival_time = departure_time + time_to_node;
                    // fix rounding errors
                    if (arrival_time > safe_int.second)
                    {
                        //TODO: Check, if this breaks everything
                        departure_time -= arrival_time - safe_int.second;
                        if(departure_time <candidate_node.first->arrival_time){
                            continue;
                        }
                        arrival_time =  safe_int.second;
                        assert(arrival_time <= safe_int.second);
                    }

                    if (arrival_time < safe_int.first)
                    {
                        // departure_time += safe_int.first - arrival_time;
                        arrival_time = safe_int.first;
                        //it's ok, because we are slowing down
                    }

                    MDP::MSIRRT::Vertex::VertexCoordType collision_coord;

                    // std::cout<<departure_time<<" "<<arrival_time<<" "<<time_to_node<<" "<<safe_int.first<<" "<<safe_int.second<<" "<<candidate_node.first->arrival_time<<" "<<candidate_node.first->safe_interval.second<<std::endl;
                    // assert(departure_time < arrival_time);
                    if (!is_collision_motion(start_coords, coord_rand, departure_time, arrival_time,collision_coord))
                    {
                        // std::cout <<departure_time<<" "<<candidate_node.first->arrival_time<<" "<<candidate_node.first->safe_interval.first<<" "<<candidate_node.first->safe_interval.second<<std::endl;
                        // assert(!is_collision_motion(start_coords, start_coords, candidate_node.first->arrival_time, departure_time));
                        // assert((safe_int.first <= arrival_time && safe_int.second >= arrival_time) || fabs(safe_int.second - arrival_time) < 0.001 || fabs(safe_int.first - arrival_time) < 0.001);
                        // assert((candidate_node.first->safe_interval.first <= departure_time && candidate_node.first->safe_interval.second >= departure_time));
                        // std::cout<<"add_vertex"<<std::endl;
                        // std::cout<<coord_rand.transpose()<<std::endl;
                        // std::cout<<safe_int.first<<" "<<safe_int.second<<std::endl;
                        // std::cout<<candidate_node.first->coords.transpose()<<std::endl;
                        // std::cout<<departure_time<<" "<<arrival_time<<std::endl;    
                        this->current_tree->add_vertex(coord_rand, safe_int, candidate_node.first, departure_time, arrival_time);
                        added_vertices.push_back(this->current_tree->array_of_vertices.back());
                        found_parent = true;
                        break;
                    }

            
                    else if((departure_time > col_coord_safe_int_second || departure_time > col_coord_safe_int_first+max_dep_time_for_safe_int_check ) && departure_time < arrival_time)
                    // else if(departure_time < arrival_time)
                       {
                            std::vector<double> vec_collision_manager(collision_coord.data(), collision_coord.data() + collision_coord.rows() * collision_coord.cols());
                            std::vector<std::pair<int, int>> collision_coord_safe_intervals = this->get_planning_safe_intervals(vec_collision_manager);
                            if (collision_coord_safe_intervals.size()==0)
                            {
                                // safe_interval_ind++;
                                this->was_static_obstacle = true;
                                break;
                            }
                            // if does not overlap, break,
                            // if overlaps, set departure time according to lowest time at safe interval
                            bool no_overlaps_found = true;
                            for (const std::pair<int, int>& collision_safe_int:collision_coord_safe_intervals){
                                if (this->cooperative_search_budget_exhausted())
                                {
                                    return added_vertices;
                                }
                                //we assume, that safe_intervals are sorted by time.
                                double time_from_candidate_to_collision =  (candidate_node.first->coords - collision_coord).norm() * (double)this->scene_task.fps / this->vmax;
                                double time_collision_to_rand =  (coord_rand - collision_coord).norm() * (double)this->scene_task.fps / this->vmax;
                                // assert(std::abs(time_from_candidate_to_collision+time_collision_to_rand  - time_to_node)<0.1 );
                                // projected to coll. coordinate parent low bound (departure time)
                                double pr_par_low = departure_time + time_from_candidate_to_collision;
                                // projected to coll. coordinate parent high bound
                                double pr_par_high = candidate_node.first->safe_interval.second + time_from_candidate_to_collision;

                                // projected to coll. coordinate random coord low bound
                                double pr_ran_low = safe_int.first - time_collision_to_rand;
                                // projected to coll. coordinate random coord high bound
                                double pr_ran_high = safe_int.second - time_collision_to_rand;

                                //ignore, if safe int is lower than dep time
                                if(collision_safe_int.second < pr_par_low){
                                    continue;
                                }
                                //stop iterating, if safe int is higher, than max dep time
                                if(collision_safe_int.first > std::min(pr_par_high,pr_ran_high)){
                                    break;
                                }

                                // get overlapping between parent and col

                                double overlap_par_col_low = std::max(pr_par_low,(double)collision_safe_int.first);
                                double overlap_par_col_high = std::min(pr_par_high,(double)collision_safe_int.second);
                                if (overlap_par_col_low > overlap_par_col_high){
                                    continue;
                                }

                                // get resulting overlapping with random coord safe_int

                                double result_overlap_low = std::max(overlap_par_col_low,pr_ran_low);
                                double result_overlap_high = std::min(overlap_par_col_high,pr_ran_high);
                                if (result_overlap_low > result_overlap_high){
                                    continue;
                                }

                                no_overlaps_found = false;

                                // std::cout<<departure_time<<" "<<candidate_node.first->safe_interval.second <<" "<<pr_par_low<<" "<<pr_par_high<<std::endl;
                                // std::cout<<safe_int.first<<" "<<safe_int.second <<" "<<pr_ran_low<<" "<<pr_ran_high<<std::endl;
                                // std::cout<<collision_safe_int.first<<" "<<collision_safe_int.second<<" "<<time_from_candidate_to_collision<<" "<<time_collision_to_rand<<std::endl;

                                // assert(result_overlap_low >=pr_par_low);
                                // std::cout<<result_overlap_low-time_from_candidate_to_collision << " "<< departure_time<<std::endl;
                                // assert(result_overlap_low-time_from_candidate_to_collision >=departure_time);
                                // std::cout<<result_overlap_low-time_from_candidate_to_collision <<" "<<result_overlap_high-time_from_candidate_to_collision<< " "<< std::min((double)candidate_node.first->safe_interval.second, (double)safe_int.second - time_to_node)<<std::endl;
                                // assert(result_overlap_low-time_from_candidate_to_collision <=std::min((double)candidate_node.first->safe_interval.second, (double)safe_int.second - time_to_node));
                                // if (!is_collision_motion(start_coords, collision_coord, departure_time, result_overlap_low))
                                // {
                                //     this->current_tree->add_vertex(collision_coord, safe_int, candidate_node.first, departure_time, result_overlap_low);
                                //     // added_vertices.push_back(this->current_tree->array_of_vertices.back());
                                // }
                                if ((result_overlap_low - time_from_candidate_to_collision - departure_time)>=1)
                                {
                                    departure_time = result_overlap_low - time_from_candidate_to_collision-1 ;
                                    col_coord_safe_int_first = departure_time+1;
                                    col_coord_safe_int_second = result_overlap_high- time_from_candidate_to_collision;
                                }

                                }
                            if (no_overlaps_found){
                                // safe_interval_ind++;
                                this->was_static_obstacle = true;
                                break;
                            }
                        }
                }
                if (found_parent)
                {
                    this->was_static_obstacle = false;
                    break;
                }
            }

            // if (!found_parent)
            // {
            //     this->orphan_tree->add_vertex(coord_rand, safe_int, nullptr, -1, safe_int.second + 1);
            // }
            // safe_interval_ind++;
        }
    }
    else if (this->current_tree == this->goal_tree)
    {

        // s-1td::cout<<"set_parent goal_tree"<<std::endl;

        double time_to_start = (coord_rand - this->root_node->coords).norm() * (double)this->scene_task.fps / this->vmax;
        // for (const std::pair<MDP::MSIRRT::Vertex *, int> &node : nearest_nodes){
        //     if (!node.first){
        //         std::cout<<"null pointer in sort!"<<std::endl;
        //         assert(false);
        //     }
        // }
        std::sort(nearest_nodes.begin(), nearest_nodes.end(), [&coord_rand,this](const std::pair<MDP::MSIRRT::Vertex *, int> &a, std::pair<MDP::MSIRRT::Vertex *, int> &b)
                  { 
                    // Add null pointer checks to prevent segfault
                    if (!a.first || !b.first) {
                        std::cout<<"null pointer in sort!"<<std::endl;
                        
                        return a.first != nullptr; // null pointers go to the end
                    }
                    return a.first->arrival_time-((coord_rand - a.first->coords).norm() * (double)this->scene_task.fps / this->vmax) > b.first->arrival_time-((coord_rand - b.first->coords).norm() * (double)this->scene_task.fps / this->vmax); 
                  });
        // std::sort(nearest_nodes.begin(), nearest_nodes.end(), [](const std::pair<MDP::MSIRRT::Vertex *, int> &a, std::pair<MDP::MSIRRT::Vertex *, int> &b)
        //     { return a.first->arrival_time > b.first->arrival_time; });

        // for (std::pair<MDP::MSIRRT::Vertex *, int> candidate_node : nearest_nodes){
        //     std::cout<<candidate_node.first->arrival_time<<", ";
        // }
        // std::cout<<std::endl;
        for (std::pair<int, int> safe_int : safe_intervals_of_coord_rand) // For each interval
        {
            if (this->cooperative_search_budget_exhausted())
            {
                return added_vertices;
            }
            // std::cout<<"safe_int: "<<safe_int.first<<" "<<safe_int.second<<std::endl;
            // if we can't reach start from that interval - skip
            if ((safe_int.second - time_to_start) < 0)
            {
                // safe_interval_ind++;
                continue;
            }
            bool found_parent = false;

            // for each parent
            for (std::pair<MDP::MSIRRT::Vertex *, int> candidate_node : nearest_nodes)
            {
                if (this->cooperative_search_budget_exhausted())
                {
                    return added_vertices;
                }
                // std::cout<<"candidate_node: "<<candidate_node.first->arrival_time<<" "<<candidate_node.first->safe_interval.first<<" "<<candidate_node.first->safe_interval.second<<std::endl;
                double time_to_node = (coord_rand - candidate_node.first->coords).norm() * (double)this->scene_task.fps / this->vmax;

                int max_dep_time_for_safe_int_check = this->max_dep_time_for_safe_int_check;
                int col_coord_safe_int_second = std::min(candidate_node.first->arrival_time, (double)safe_int.second + time_to_node)-max_dep_time_for_safe_int_check;
                int col_coord_safe_int_first = col_coord_safe_int_second-1;

                if (time_to_node<1){ // duplicate... this point was already added
                    found_parent = true;
                    break;
                }
                // candidate nodes are sorted by descending arrival time. If arrival time of parent < safe int bound + time to node - > break;
                if (candidate_node.first->arrival_time - time_to_node < safe_int.first)
                {
                    break;
                }

                // if intervals don't overlap - skip
                if ((candidate_node.first->safe_interval.first - time_to_node > safe_int.second))
                {
                    continue;
                }
                // assert(candidate_node.first);

                MDP::MSIRRT::Vertex::VertexCoordType start_coords = candidate_node.first->coords;

                // for each departure time
                for (double departure_time = std::min(candidate_node.first->arrival_time, (double)safe_int.second + time_to_node); departure_time >= std::max((double)candidate_node.first->safe_interval.first, (double)safe_int.first + time_to_node); departure_time -= 1)
                {
                    if (this->cooperative_search_budget_exhausted())
                    {
                        return added_vertices;
                    }
                    // std::cout<<"departure_time: "<<departure_time<<std::endl;
                    // assert(!is_collision_motion(candidate_node.first->coords, candidate_node.first->coords, candidate_node.first->arrival_time, departure_time));
                    double arrival_time = departure_time - time_to_node; // we travel to the past
                    // fix rounding errors
                    if (arrival_time > safe_int.second)
                    {
                        // departure_time -= arrival_time - safe_int.second;
                        arrival_time = safe_int.second;
                    }
                    if (arrival_time < safe_int.first)
                    {
                        departure_time += safe_int.first - arrival_time;
                        if(departure_time < candidate_node.first->arrival_time){
                            continue;
                        }
                        arrival_time = departure_time - time_to_node;
                        assert(arrival_time >= safe_int.first);
                    }
                    MDP::MSIRRT::Vertex::VertexCoordType collision_coord;
                    // assert(departure_time > arrival_time);
                    if (!is_collision_motion(coord_rand, start_coords, arrival_time, departure_time,collision_coord))
                    {
                        // assert(!is_collision_motion(start_coords, start_coords, departure_time, candidate_node.first->arrival_time));
                        // std::cout<<safe_int.first<<" "<<safe_int.second<<" "<<arrival_time<<std::endl;
                        // assert((safe_int.first <= arrival_time && safe_int.second >= arrival_time) || fabs(safe_int.second - arrival_time) < 0.001 || fabs(safe_int.first - arrival_time) < 0.001);
                        // std::cout<<"add_vertex"<<std::endl;
                        // std::cout<<coord_rand.transpose()<<std::endl;
                        // std::cout<<safe_int.first<<" "<<safe_int.second<<std::endl;
                        // std::cout<<candidate_node.first->coords.transpose()<<std::endl;
                        // std::cout<<departure_time<<" "<<arrival_time<<std::endl;    
                        this->current_tree->add_vertex(coord_rand, safe_int, candidate_node.first, departure_time, arrival_time);
                        added_vertices.push_back(this->current_tree->array_of_vertices.back());
                        found_parent = true;
                        break;
                    }

                    else if((departure_time < col_coord_safe_int_first || departure_time < col_coord_safe_int_second-max_dep_time_for_safe_int_check ) && departure_time > arrival_time)
                    // else if(departure_time > arrival_time)
                        {
                            
                            std::vector<std::pair<int, int>> collision_coord_safe_intervals = this->get_planning_safe_intervals(std::vector<double>(collision_coord.data(), collision_coord.data() + collision_coord.rows() * collision_coord.cols()));
                            if (collision_coord_safe_intervals.size()==0)
                            {
                                this->was_static_obstacle = true;
                                break;
                            }
                            // if does not overlap, break,
                            // if overlaps, set departure time according to lowest time at safe interval
                            bool no_overlaps_found = true;

                            // reverse vector, so it sorted from latest to earliest safe intervals
                            std::reverse(collision_coord_safe_intervals.begin(), collision_coord_safe_intervals.end());

                            for (const std::pair<int, int>& collision_safe_int:collision_coord_safe_intervals){
                                if (this->cooperative_search_budget_exhausted())
                                {
                                    return added_vertices;
                                }
                                //we assume, that safe_intervals are sorted by time from latest to earliest.
                                double time_from_candidate_to_collision =  (candidate_node.first->coords - collision_coord).norm() * (double)this->scene_task.fps / this->vmax;
                                double time_collision_to_rand =  (coord_rand - collision_coord).norm() * (double)this->scene_task.fps / this->vmax;

                                // projected to coll. coordinate parent low bound
                                double pr_par_low = candidate_node.first->safe_interval.first - time_from_candidate_to_collision;
                                // projected to coll. coordinate parent high bound (departure time)
                                double pr_par_high = departure_time - time_from_candidate_to_collision;

                                // projected to coll. coordinate random coord low bound
                                double pr_ran_low = safe_int.first + time_collision_to_rand;
                                // projected to coll. coordinate random coord high bound
                                double pr_ran_high = safe_int.second + time_collision_to_rand;

                                //ignore, if safe int is higher than dep time
                                if(collision_safe_int.first > pr_par_high){
                                    continue;
                                }
                                //stop iterating, if safe int is lower, than min dep time
                                if(collision_safe_int.second < std::min(pr_par_low,pr_ran_low)){
                                    break;
                                }

                                // get overlapping between parent and col

                                double overlap_par_col_low = std::max(pr_par_low,(double)collision_safe_int.first);
                                double overlap_par_col_high = std::min(pr_par_high,(double)collision_safe_int.second);
                                if (overlap_par_col_low > overlap_par_col_high){
                                    continue;
                                }

                                // get resulting overlapping with random coord safe_int

                                double result_overlap_low = std::max(overlap_par_col_low,pr_ran_low);
                                double result_overlap_high = std::min(overlap_par_col_high,pr_ran_high);
                                if (result_overlap_low > result_overlap_high){
                                    continue;
                                }

                                no_overlaps_found = false;
                                // if (!is_collision_motion(collision_coord, start_coords , result_overlap_high, departure_time))
                                // {
                                //     this->current_tree->add_vertex(collision_coord, safe_int, candidate_node.first, departure_time, result_overlap_high);
                                //     // added_vertices.push_back(this->current_tree->array_of_vertices.back());
                                // }
                                // assert(result_overlap_high <= pr_par_high);
                                // assert(result_overlap_high + time_from_candidate_to_collision <=departure_time);
                                // assert(result_overlap_high + time_from_candidate_to_collision >= std::max((double)candidate_node.first->safe_interval.first, (double)safe_int.first + time_to_node));
                                if (( departure_time-(result_overlap_high + time_from_candidate_to_collision))>1)
                                {
                                    departure_time = result_overlap_high + time_from_candidate_to_collision+1 ;
                                    col_coord_safe_int_first = result_overlap_low + time_from_candidate_to_collision;
                                    col_coord_safe_int_second = departure_time -1;
                                }

                                }
                            if (no_overlaps_found){
                            //    safe_interval_ind++;
                            this->was_static_obstacle = true;   
                                break;
                            }
                        }
                }
                if (found_parent)
                {
                    break;
                }
            }

            // if (!found_parent)
            // {
            //     this->orphan_tree->add_vertex(coord_rand, safe_int, nullptr, -1, safe_int.second + 1);
            // }
            // safe_interval_ind++;
        }
    }

    return added_vertices;
}

bool MDP::MSIRRT::PlannerConnect::is_goal(const MDP::MSIRRT::Vertex::VertexCoordType &coord)
{
    MDP::MSIRRT::Vertex::VertexCoordType delta = this->goal_coords - coord;
    double delta_norm = delta.norm();
    return delta_norm < 0.01;
}

bool MDP::MSIRRT::PlannerConnect::is_collision_state(MDP::MSIRRT::Vertex::VertexCoordType &coords, int &time)
{
    std::vector<double> robot_angles;
    for (int joint_int = 0; joint_int < coords.size(); joint_int++)
    {
        robot_angles.push_back((double)coords[joint_int]);
    }
    return this->collision_manager->check_collision_frame(robot_angles, time);
}

bool MDP::MSIRRT::PlannerConnect::is_collision_motion(const MDP::MSIRRT::Vertex::VertexCoordType &start_coords, const MDP::MSIRRT::Vertex::VertexCoordType &end_coords, double &start_time, double &end_time, MDP::MSIRRT::Vertex::VertexCoordType &collision_coord)
{
    last_valid_coord = start_coords;
    if (start_time >= end_time)
    {
        std::cout << "start_time > end_time" << std::endl;

        return true;
    }
    MDP::MSIRRT::Vertex::VertexCoordType dir_vector = end_coords - start_coords;
    if ((dir_vector.norm() * ((double)this->scene_task.fps) / ((double)(end_time - start_time))) > (this->vmax+0.001)) // We need 0.001 to fix floating point comparison errors.
    {
        std::cout << "((dir_vector.norm() * ((double)this->scene_task.fps) / ((double)(end_time - start_time))) >= this->vmax)" << std::endl;
        std::cout <<std::fixed <<std::setprecision(15) <<(dir_vector.norm() * ((double)this->scene_task.fps) / ((double)(end_time - start_time)))<< " "<< dir_vector.norm() << " " << (double)this->scene_task.fps << " " << ((double)(end_time - start_time)) << " "<<std::fixed <<std::setprecision(15)<< this->vmax  << std::endl;
        return true;
    }

    int interpolation_steps = std::max({(int)(dir_vector.norm() / 0.1), (int)(end_time - start_time + 0.5), 1});

    // std::cout<<"interpolation_steps: "<<interpolation_steps<<" "<< dir_vector.norm()<<std::endl;
    for (const auto &sample : MDP::MSIRRT::motion_validation_samples(
             start_time, end_time, interpolation_steps))
    {
        if (this->cooperative_search_budget_exhausted())
        {
            collision_coord = last_valid_coord;
            return true;
        }
        MDP::MSIRRT::Vertex::VertexCoordType temp_coords = start_coords + dir_vector * sample.alpha;
        int time_frame = sample.frame;

        if (this->is_collision_state(temp_coords, time_frame))
        {
            // std::cout << "is_collision!!" << std::endl;
            collision_coord = temp_coords;
            return true;
        }
        last_valid_coord = temp_coords;
    }
    return false;
}

bool MDP::MSIRRT::PlannerConnect::is_collision_motion(const MDP::MSIRRT::Vertex::VertexCoordType &start_coords, const MDP::MSIRRT::Vertex::VertexCoordType &end_coords, double &start_time, double &end_time)
{

    if (start_time >= end_time)
    {
        // std::cout << "start_time > end_time" << std::endl;
        return true;
    }
    MDP::MSIRRT::Vertex::VertexCoordType dir_vector = end_coords - start_coords;
    if ((dir_vector.norm() * ((double)this->scene_task.fps) / ((double)(end_time - start_time))) > (this->vmax+0.001)) // We need 0.001 to fix floating point comparison errors.
    {
        std::cout << "((dir_vector.norm() * ((double)this->scene_task.fps) / ((double)(end_time - start_time))) >= this->vmax)" << std::endl;
        std::cout <<std::fixed <<std::setprecision(15) <<(dir_vector.norm() * ((double)this->scene_task.fps) / ((double)(end_time - start_time)))<< " "<< dir_vector.norm() << " " << (double)this->scene_task.fps << " " << ((double)(end_time - start_time)) << " "<<std::fixed <<std::setprecision(15)<< this->vmax  << std::endl;
        return true;
    }

    int interpolation_steps = std::max({(int)(dir_vector.norm() / 0.1), (int)(end_time - start_time + 0.5), 1});

    // std::cout<<"interpolation_steps: "<<interpolation_steps<<" "<< dir_vector.norm()<<std::endl;
    for (const auto &sample : MDP::MSIRRT::motion_validation_samples(
             start_time, end_time, interpolation_steps))
    {
        if (this->cooperative_search_budget_exhausted())
        {
            return true;
        }
        MDP::MSIRRT::Vertex::VertexCoordType temp_coords = start_coords + dir_vector * sample.alpha;
        int time_frame = sample.frame;

        if (this->is_collision_state(temp_coords, time_frame))
        {
            // std::cout << "is_collision!!" << std::endl;

            return true;
        }
    }
    return false;
}

bool MDP::MSIRRT::PlannerConnect::check_path(std::vector<MDP::MSIRRT::Vertex *> &path)
{
    if (path.empty())
    {
        return false;
    }
    for (const auto *vertex : path)
    {
        if (vertex == nullptr || !vertex->active ||
            vertex->arrival_time < vertex->safe_interval.first ||
            vertex->arrival_time > vertex->safe_interval.second ||
            vertex->arrival_time < this->planning_start_frame ||
            vertex->arrival_time > this->planning_horizon_frame)
        {
            return false;
        }
    }
    for (std::size_t node_id = 0; node_id + 1 < path.size(); ++node_id)
    {
        auto *parent = path[node_id];
        auto *child = path[node_id + 1];
        if (child->parent != parent || child->departure_from_parent_time < parent->arrival_time ||
            child->departure_from_parent_time > parent->safe_interval.second ||
            child->arrival_time <= child->departure_from_parent_time)
        {
            return false;
        }
        double wait_start = parent->arrival_time;
        double wait_end = child->departure_from_parent_time;
        if (wait_end > wait_start &&
            this->is_collision_motion(parent->coords, parent->coords, wait_start, wait_end))
        {
            return false;
        }
        double motion_start = child->departure_from_parent_time;
        double motion_end = child->arrival_time;
        if (this->is_collision_motion(parent->coords, child->coords, motion_start, motion_end))
        {
            return false;
        }
    }
    return true;
}

void MDP::MSIRRT::PlannerConnect::prune_goal_tree()
{
    assert(this->goal_reached);

    MDP::MSIRRT::Vertex *start_tree_node = this->goal_nodes.first;
    MDP::MSIRRT::Vertex *goal_tree_node_child = this->goal_nodes.second;

    assert(goal_tree_node_child->parent); // Must have parent, because RRTConnect can't just connect to root node
    MDP::MSIRRT::Vertex *goal_tree_node = goal_tree_node_child->parent;

    while (goal_tree_node)
    {
        double time_to_node = (start_tree_node->coords - goal_tree_node->coords).norm() * (double)this->scene_task.fps / this->vmax;

        MDP::MSIRRT::Vertex::VertexCoordType start_coords = start_tree_node->coords;
        bool found_dep_time = false;
        // for each departure time
        for (double departure_time = std::max(start_tree_node->arrival_time, (double)goal_tree_node->safe_interval.first - time_to_node); departure_time <= goal_tree_node_child->arrival_time; departure_time += 1)
        {

            // assert(!is_collision_motion(start_coords, start_coords, start_tree_node->arrival_time, departure_time));
            double arrival_time = departure_time + time_to_node;
            if (!is_collision_motion(start_coords, goal_tree_node->coords, departure_time, arrival_time))
            {
                this->start_tree->add_vertex(goal_tree_node->coords, goal_tree_node->safe_interval, start_tree_node, departure_time, arrival_time);
                start_tree_node = this->start_tree->array_of_vertices.back();
                found_dep_time = true;
                break;
            }
        }
        if (!found_dep_time)
        {
            this->start_tree->add_vertex(goal_tree_node->coords, goal_tree_node->safe_interval, start_tree_node, goal_tree_node_child->arrival_time, goal_tree_node_child->departure_from_parent_time);
            start_tree_node = this->start_tree->array_of_vertices.back();
        }
        goal_tree_node_child = goal_tree_node;
        goal_tree_node = goal_tree_node->parent;
    }
    this->finish_node = start_tree_node;
}


int MDP::MSIRRT::PlannerConnect::get_start_tree_vertices_count() const
{
    return this->start_tree->array_of_vertices.size();
}

int MDP::MSIRRT::PlannerConnect::get_goal_tree_vertices_count() const
{
    return this->goal_tree->array_of_vertices.size();
}

std::size_t MDP::MSIRRT::PlannerConnect::get_start_tree_active_vertices_count() const
{
    return this->start_tree->active_vertex_count();
}

std::size_t MDP::MSIRRT::PlannerConnect::get_goal_tree_active_vertices_count() const
{
    return this->goal_tree->active_vertex_count();
}

std::vector<MDP::MSIRRT::FrameRange> MDP::MSIRRT::PlannerConnect::prediction_change_windows(
    const MDP::ConfigReader::SceneTask &updated_scene_task) const
{
    if (updated_scene_task.frame_count != this->scene_task.frame_count ||
        updated_scene_task.fps != this->scene_task.fps)
    {
        throw std::invalid_argument("prediction update must preserve frame_count and fps");
    }
    if (updated_scene_task.start_configuration != this->scene_task.start_configuration ||
        updated_scene_task.end_configuration != this->scene_task.end_configuration)
    {
        throw std::invalid_argument(
            "prediction update must preserve the current start and goal configurations; advance the start first");
    }
    if (updated_scene_task.obstacles.size() != this->scene_task.obstacles.size())
    {
        throw std::invalid_argument("prediction update must preserve obstacle count");
    }
    if (!updated_scene_task.robot_obstacles.empty() || !this->scene_task.robot_obstacles.empty())
    {
        throw std::invalid_argument("incremental prediction updates currently support scene obstacles, not robot obstacles");
    }

    std::vector<MDP::MSIRRT::FrameRange> changed;
    for (std::size_t obstacle_id = 0; obstacle_id < this->scene_task.obstacles.size(); ++obstacle_id)
    {
        const auto &before_obstacle = this->scene_task.obstacles[obstacle_id];
        const auto &after_obstacle = updated_scene_task.obstacles[obstacle_id];
        if (before_obstacle->get_name() != after_obstacle->get_name() ||
            before_obstacle->get_type() != after_obstacle->get_type() ||
            before_obstacle->get_obstacle_type() != after_obstacle->get_obstacle_type())
        {
            throw std::invalid_argument("prediction update must preserve obstacle identity and geometry type");
        }

        const auto &before_coordinates = before_obstacle->get_coordinates();
        const auto &after_coordinates = after_obstacle->get_coordinates();
        const std::size_t before_minimum = before_obstacle->get_is_static() ? 1 : this->scene_task.frame_count;
        const std::size_t after_minimum = after_obstacle->get_is_static() ? 1 : updated_scene_task.frame_count;
        if (before_coordinates.size() < before_minimum || after_coordinates.size() < after_minimum)
        {
            throw std::invalid_argument("prediction obstacle trajectory does not cover frame_count");
        }

        const MDP::SphereObstacleJsonInfo *before_sphere = nullptr;
        const MDP::SphereObstacleJsonInfo *after_sphere = nullptr;
        if (before_obstacle->get_obstacle_type() == MDP::ObstacleJsonInfo::ObstacleType::SPHERE)
        {
            before_sphere = static_cast<const MDP::SphereObstacleJsonInfo *>(before_obstacle.get());
            after_sphere = static_cast<const MDP::SphereObstacleJsonInfo *>(after_obstacle.get());
        }

        std::vector<MDP::MSIRRT::PredictionSample> before_samples;
        std::vector<MDP::MSIRRT::PredictionSample> after_samples;
        before_samples.reserve(this->scene_task.frame_count);
        after_samples.reserve(this->scene_task.frame_count);
        for (std::size_t frame = 0; frame < this->scene_task.frame_count; ++frame)
        {
            const auto &before_position = before_coordinates[before_obstacle->get_is_static() ? 0 : frame];
            const auto &after_position = after_coordinates[after_obstacle->get_is_static() ? 0 : frame];

            MDP::MSIRRT::PredictionSample before_sample;
            before_sample.center = {before_position.x, before_position.y, before_position.z};
            MDP::MSIRRT::PredictionSample after_sample;
            after_sample.center = {after_position.x, after_position.y, after_position.z};
            if (before_sphere != nullptr)
            {
                before_sample.tube_radius = before_sphere->get_radius() + before_sphere->get_uncertainty_radius(frame);
                after_sample.tube_radius = after_sphere->get_radius() + after_sphere->get_uncertainty_radius(frame);
            }
            before_samples.push_back(before_sample);
            after_samples.push_back(after_sample);
        }

        const auto obstacle_changes = MDP::MSIRRT::prediction_change_windows(
            before_samples, after_samples, 1e-9);
        changed.insert(changed.end(), obstacle_changes.begin(), obstacle_changes.end());
    }
    if (updated_scene_task.reliable_until_frame != this->scene_task.reliable_until_frame)
    {
        const int first = static_cast<int>(std::min(
            updated_scene_task.reliable_until_frame,
            this->scene_task.reliable_until_frame)) + 1;
        const int last = static_cast<int>(std::max(
            updated_scene_task.reliable_until_frame,
            this->scene_task.reliable_until_frame));
        if (first <= last)
        {
            changed.emplace_back(first, last);
        }
    }
    const int relevant_last_frame = std::min(
        static_cast<int>(this->scene_task.frame_count) - 1,
        std::max(
            this->planning_horizon_frame,
            static_cast<int>(updated_scene_task.reliable_until_frame)));
    std::vector<MDP::MSIRRT::FrameRange> relevant_changes;
    const int relevant_first_frame = this->executed_prefix ? this->planning_start_frame : 0;
    for (const auto &range : MDP::MSIRRT::normalize_frame_ranges(changed))
    {
        const int first = std::max(relevant_first_frame, range.first);
        const int last = std::min(relevant_last_frame, range.last);
        if (first <= last)
        {
            relevant_changes.emplace_back(first, last);
        }
    }
    return relevant_changes;
}

bool MDP::MSIRRT::PlannerConnect::is_tree_edge_valid(
    const MDP::MSIRRT::Tree *tree,
    const MDP::MSIRRT::Vertex &parent,
    const MDP::MSIRRT::Vertex &child)
{
    if (!parent.active || !child.active || child.parent != &parent)
    {
        return false;
    }

    if (tree == this->start_tree)
    {
        double wait_start = parent.arrival_time;
        double wait_end = child.departure_from_parent_time;
        if (wait_end < wait_start)
        {
            return false;
        }
        if (wait_end > wait_start && this->is_collision_motion(parent.coords, parent.coords, wait_start, wait_end))
        {
            return false;
        }

        double motion_start = child.departure_from_parent_time;
        double motion_end = child.arrival_time;
        return motion_end > motion_start && !this->is_collision_motion(parent.coords, child.coords, motion_start, motion_end);
    }

    double motion_start = child.arrival_time;
    double motion_end = child.departure_from_parent_time;
    if (motion_end <= motion_start || this->is_collision_motion(child.coords, parent.coords, motion_start, motion_end))
    {
        return false;
    }
    double wait_start = child.departure_from_parent_time;
    double wait_end = parent.arrival_time;
    return wait_end >= wait_start &&
           (wait_end == wait_start || !this->is_collision_motion(parent.coords, parent.coords, wait_start, wait_end));
}

std::size_t MDP::MSIRRT::PlannerConnect::reconnect_invalidated_vertices(
    MDP::MSIRRT::Tree *tree,
    const std::vector<MDP::MSIRRT::Vertex *> &invalidated_vertices)
{
    if (this->root_node == nullptr || !this->root_node->active)
    {
        return 0;
    }

    std::vector<MDP::MSIRRT::Vertex *> ordered = invalidated_vertices;
    auto depth = [](const MDP::MSIRRT::Vertex *vertex) {
        std::size_t value = 0;
        while (vertex->parent != nullptr)
        {
            ++value;
            vertex = vertex->parent;
        }
        return value;
    };
    std::stable_sort(ordered.begin(), ordered.end(), [&](const auto *left, const auto *right) {
        return depth(left) < depth(right);
    });

    MDP::MSIRRT::Tree *saved_current = this->current_tree;
    MDP::MSIRRT::Tree *saved_other = this->other_tree;
    this->current_tree = tree;
    this->other_tree = tree == this->start_tree ? this->goal_tree : this->start_tree;

    std::size_t reconnected = 0;
    for (MDP::MSIRRT::Vertex *invalidated : ordered)
    {
        if (invalidated->parent == nullptr || tree->active_vertex_count() == 0)
        {
            continue;
        }
        auto coords = invalidated->coords;
        const std::vector<double> robot_angles(coords.data(), coords.data() + coords.size());
        auto safe_intervals = this->get_planning_safe_intervals(robot_angles);
        const auto added = this->set_parent(coords, safe_intervals);
        reconnected += added.size();
    }

    this->current_tree = saved_current;
    this->other_tree = saved_other;
    return reconnected;
}

void MDP::MSIRRT::PlannerConnect::ensure_roots_after_prediction_update()
{
    const auto start_intervals = this->get_planning_safe_intervals(this->scene_task.start_configuration);
    const auto start_interval = std::find_if(start_intervals.begin(), start_intervals.end(), [&](const auto &interval) {
        return interval.first <= this->planning_start_frame && this->planning_start_frame <= interval.second;
    });
    this->root_node = nullptr;
    for (MDP::MSIRRT::Vertex *vertex : this->start_tree->array_of_vertices)
    {
        if (vertex->active && vertex->parent == nullptr &&
            std::abs(vertex->arrival_time - this->planning_start_frame) < 1e-9)
        {
            this->root_node = vertex;
            break;
        }
    }
    if (this->root_node == nullptr && start_interval != start_intervals.end())
    {
        this->start_tree->add_vertex(
            this->scene_task.start_configuration,
            *start_interval,
            nullptr,
            -1,
            this->planning_start_frame);
        this->root_node = this->start_tree->array_of_vertices.back();
    }

    this->goal_safe_intervals = this->get_planning_safe_intervals(this->scene_task.end_configuration);
    for (const auto &interval : this->goal_safe_intervals)
    {
        const bool represented = std::any_of(
            this->goal_tree->array_of_vertices.begin(),
            this->goal_tree->array_of_vertices.end(),
            [&](const MDP::MSIRRT::Vertex *vertex) {
                return vertex->active && vertex->parent == nullptr && vertex->safe_interval == interval;
            });
        if (!represented)
        {
            this->goal_tree->add_vertex(this->goal_coords, interval, nullptr, -1, interval.second);
        }
    }
}

MDP::MSIRRT::BidirectionalRepairReport MDP::MSIRRT::PlannerConnect::update_prediction(
    MDP::ConfigReader::SceneTask updated_scene_task)
{
    MDP::MSIRRT::BidirectionalRepairReport result;
    auto previous_path = this->get_final_path();
    const bool had_previous_solution = !previous_path.empty();
    const auto tube_update_start = std::chrono::steady_clock::now();
    result.changed_windows = this->prediction_change_windows(updated_scene_task);
    result.tube_update_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - tube_update_start).count();
    result.previous_solution_invalidated = had_previous_solution && !result.changed_windows.empty();
    result.prediction_version = ++this->prediction_version;

    using SafeIntervalSnapshot = std::unordered_map<
        std::size_t,
        std::vector<std::pair<int, int>>>;
    const auto snapshot_safe_intervals = [&](const MDP::MSIRRT::Tree *tree) {
        SafeIntervalSnapshot snapshot;
        if (this->reuse_stored_previous_intervals || result.changed_windows.empty())
        {
            return snapshot;
        }
        const auto affected = tree->affected_by(result.changed_windows);
        for (const std::size_t vertex_id : affected.vertex_ids)
        {
            if (vertex_id >= tree->array_of_vertices.size())
            {
                continue;
            }
            const auto *vertex = tree->array_of_vertices[vertex_id];
            if (!vertex->active)
            {
                continue;
            }
            snapshot.emplace(
                vertex_id,
                this->get_planning_safe_intervals(std::vector<double>(
                    vertex->coords.data(),
                    vertex->coords.data() + vertex->coords.size()),
                    &result.safe_interval_update_seconds));
        }
        return snapshot;
    };
    const auto start_previous_safe_intervals = snapshot_safe_intervals(this->start_tree);
    const auto goal_previous_safe_intervals = snapshot_safe_intervals(this->goal_tree);

    if (!result.changed_windows.empty())
    {
        const auto collision_start = std::chrono::steady_clock::now();
        std::vector<std::pair<int, int>> collision_windows;
        collision_windows.reserve(result.changed_windows.size());
        for (const auto &window : result.changed_windows)
        {
            collision_windows.emplace_back(window.first, window.last);
        }
        const auto collision_report = this->horizon_bounded_safe_interval_queries_enabled
            ? this->collision_manager->update_scene(
                  updated_scene_task,
                  collision_windows,
                  std::min(
                      static_cast<int>(updated_scene_task.reliable_until_frame),
                      static_cast<int>(updated_scene_task.frame_count) - 1))
            : this->collision_manager->update_scene(
                  updated_scene_task, collision_windows);
        result.collision_frames_rebuilt = collision_report.frames_rebuilt;
        result.collision_obstacles_reindexed = collision_report.obstacles_reindexed;
        result.collision_full_rebuild_fallback = collision_report.full_rebuild_fallback;
        this->scene_task = std::move(updated_scene_task);
        if (collision_report.full_rebuild_fallback)
        {
            this->collision_manager = std::make_unique<MDP::CollisionManager>(this->scene_task);
        }
        result.collision_structure_update_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - collision_start).count();
        result.safe_interval_update_seconds += result.collision_structure_update_seconds;
        this->goal_reached = false;
        this->goal_nodes = {nullptr, nullptr};
        this->finish_node = nullptr;
    }
    else
    {
        this->scene_task = std::move(updated_scene_task);
    }
    this->planning_horizon_frame = static_cast<int>(std::min(
        this->scene_task.reliable_until_frame,
        this->scene_task.frame_count - 1));

    const auto repair_start = std::chrono::steady_clock::now();
    double repair_safe_interval_query_seconds = 0.0;
    const auto current_safe_interval_query = [&](const MDP::MSIRRT::Vertex &vertex) {
        return this->get_planning_safe_intervals(
            std::vector<double>(vertex.coords.data(), vertex.coords.data() + vertex.coords.size()),
            &repair_safe_interval_query_seconds);
    };
    const auto previous_query = [](const SafeIntervalSnapshot &snapshot) {
        return [&snapshot](const MDP::MSIRRT::Vertex &vertex) {
            const auto found = snapshot.find(vertex.vertex_id);
            if (found != snapshot.end())
            {
                return found->second;
            }
            return std::vector<std::pair<int, int>>{vertex.safe_interval};
        };
    };
    const MDP::MSIRRT::SafeIntervalQuery start_previous_query =
        this->reuse_stored_previous_intervals
            ? MDP::MSIRRT::SafeIntervalQuery{}
            : previous_query(start_previous_safe_intervals);
    const MDP::MSIRRT::SafeIntervalQuery goal_previous_query =
        this->reuse_stored_previous_intervals
            ? MDP::MSIRRT::SafeIntervalQuery{}
            : previous_query(goal_previous_safe_intervals);
    result.used_stored_previous_intervals =
        this->reuse_stored_previous_intervals && !result.changed_windows.empty();
    auto start_outcome = MDP::MSIRRT::repair_tree(
        *this->start_tree,
        result.changed_windows,
        this->prediction_version,
        start_previous_query,
        current_safe_interval_query,
        [&](const auto &parent, const auto &child) {
            return this->is_tree_edge_valid(this->start_tree, parent, child);
        });
    auto goal_outcome = MDP::MSIRRT::repair_tree(
        *this->goal_tree,
        result.changed_windows,
        this->prediction_version,
        goal_previous_query,
        current_safe_interval_query,
        [&](const auto &parent, const auto &child) {
            return this->is_tree_edge_valid(this->goal_tree, parent, child);
        });

    this->ensure_roots_after_prediction_update();
    start_outcome.report.reconnected_vertices = this->reconnect_invalidated_vertices(
        this->start_tree, start_outcome.invalidated_vertices);
    goal_outcome.report.reconnected_vertices = this->reconnect_invalidated_vertices(
        this->goal_tree, goal_outcome.invalidated_vertices);
    start_outcome.report.invariant_error = MDP::MSIRRT::describe_tree_invariant_violation(*this->start_tree);
    goal_outcome.report.invariant_error = MDP::MSIRRT::describe_tree_invariant_violation(*this->goal_tree);
    start_outcome.report.invariants_hold = start_outcome.report.invariant_error.empty();
    goal_outcome.report.invariants_hold = goal_outcome.report.invariant_error.empty();

    result.start_tree = start_outcome.report;
    result.goal_tree = goal_outcome.report;
    result.invariants_hold = result.start_tree.invariants_hold && result.goal_tree.invariants_hold;
    const bool previous_path_preserved = had_previous_solution &&
        !previous_path.empty() && previous_path.front() == this->root_node &&
        this->is_goal(previous_path.back()->coords) && this->check_path(previous_path);
    if (previous_path_preserved)
    {
        this->goal_reached = true;
        this->finish_node = previous_path.back();
        result.previous_solution_invalidated = false;
        result.previous_solution_preserved = true;
    }
    result.repair_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - repair_start).count();
    result.safe_interval_update_seconds += repair_safe_interval_query_seconds;
    const double exclusive_tree_repair_seconds =
        result.repair_seconds - repair_safe_interval_query_seconds;
    if (exclusive_tree_repair_seconds < -1e-9)
    {
        throw std::runtime_error("safe-interval timing exceeds the complete repair block");
    }
    result.tree_repair_seconds = std::max(0.0, exclusive_tree_repair_seconds);
    return result;
}

std::size_t MDP::MSIRRT::PlannerConnect::get_prediction_version() const
{
    return this->prediction_version;
}

std::vector<std::pair<int, int>> MDP::MSIRRT::PlannerConnect::clip_safe_intervals_to_horizon(
    const std::vector<std::pair<int, int>> &safe_intervals) const
{
    std::vector<std::pair<int, int>> result;
    for (const auto &interval : safe_intervals)
    {
        if (interval.first > this->planning_horizon_frame)
        {
            break;
        }
        const int first = std::max(interval.first, this->planning_start_frame);
        const int last = std::min(interval.second, this->planning_horizon_frame);
        if (first <= last)
        {
            result.emplace_back(first, last);
        }
    }
    return result;
}

std::vector<std::pair<int, int>> MDP::MSIRRT::PlannerConnect::get_planning_safe_intervals(
    const std::vector<double> &robot_angles,
    double *elapsed_seconds)
{
    const auto start = std::chrono::steady_clock::now();
    const auto raw_intervals = this->horizon_bounded_safe_interval_queries_enabled
        ? this->collision_manager->get_safe_intervals(robot_angles, this->planning_horizon_frame)
        : this->collision_manager->get_safe_intervals(robot_angles);
    auto result = this->clip_safe_intervals_to_horizon(
        raw_intervals);
    if (elapsed_seconds != nullptr)
    {
        *elapsed_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    }
    return result;
}

bool MDP::MSIRRT::PlannerConnect::get_horizon_bounded_safe_interval_queries_enabled() const
{
    return this->horizon_bounded_safe_interval_queries_enabled;
}

std::vector<MDP::MSIRRT::Vertex *> MDP::MSIRRT::PlannerConnect::nearest_active_vertices(
    MDP::MSIRRT::Tree *tree,
    const MDP::MSIRRT::Vertex::VertexCoordType &coords,
    std::size_t count)
{
    std::vector<MDP::MSIRRT::Vertex *> result;
    const std::size_t available = tree->active_vertex_count();
    const std::size_t requested = std::min(count, available);
    if (requested == 0)
    {
        return result;
    }
    std::vector<std::size_t> indices(requested);
    std::vector<double> distances(requested);
    nanoflann::KNNResultSet<double> result_set(requested);
    result_set.init(indices.data(), distances.data());
    tree->kd_tree.findNeighbors(result_set, coords.data(), {0});
    result.reserve(requested);
    for (std::size_t index = 0; index < result_set.size(); ++index)
    {
        MDP::MSIRRT::Vertex *vertex = tree->array_of_vertices[indices[index]];
        if (vertex->active)
        {
            result.push_back(vertex);
        }
    }
    return result;
}

bool MDP::MSIRRT::PlannerConnect::is_sample_temporally_reachable(
    const MDP::MSIRRT::Vertex::VertexCoordType &coords,
    const std::vector<std::pair<int, int>> &safe_intervals)
{
    const auto bounds = this->tree_conditioned_time_bounds(coords);
    const double earliest_from_start = bounds.first;
    const double latest_for_goal = bounds.second;

    for (const auto &interval : safe_intervals)
    {
        const double earliest = std::max(earliest_from_start, static_cast<double>(interval.first));
        const double latest = std::min(latest_for_goal, static_cast<double>(interval.second));
        if (earliest <= latest)
        {
            return true;
        }
    }
    return false;
}

bool MDP::MSIRRT::PlannerConnect::is_sample_kinematically_reachable(
    const MDP::MSIRRT::Vertex::VertexCoordType &coords)
{
    const auto bounds = this->tree_conditioned_time_bounds(coords);
    return bounds.first <= bounds.second;
}

std::pair<double, double> MDP::MSIRRT::PlannerConnect::tree_conditioned_time_bounds(
    const MDP::MSIRRT::Vertex::VertexCoordType &coords)
{
    const MDP::MSIRRT::Vertex::VertexCoordType start_coords(this->scene_task.start_configuration.data());
    double earliest_from_start = static_cast<double>(this->planning_start_frame) + (coords - start_coords).norm() *
        static_cast<double>(this->scene_task.fps) / this->vmax;
    double latest_for_goal = static_cast<double>(this->planning_horizon_frame) -
        (this->goal_coords - coords).norm() * static_cast<double>(this->scene_task.fps) / this->vmax;

    const auto start_anchors = this->nearest_active_vertices(this->start_tree, coords, 8);
    if (!start_anchors.empty())
    {
        double tree_earliest = std::numeric_limits<double>::infinity();
        for (const auto *anchor : start_anchors)
        {
            tree_earliest = std::min(
                tree_earliest,
                anchor->arrival_time + (coords - anchor->coords).norm() *
                    static_cast<double>(this->scene_task.fps) / this->vmax);
        }
        earliest_from_start = std::max(earliest_from_start, tree_earliest);
    }

    const auto goal_anchors = this->nearest_active_vertices(this->goal_tree, coords, 8);
    if (!goal_anchors.empty())
    {
        double tree_latest = -std::numeric_limits<double>::infinity();
        for (const auto *anchor : goal_anchors)
        {
            tree_latest = std::max(
                tree_latest,
                anchor->arrival_time - (coords - anchor->coords).norm() *
                    static_cast<double>(this->scene_task.fps) / this->vmax);
        }
        latest_for_goal = std::min(latest_for_goal, tree_latest);
    }

    return {earliest_from_start, latest_for_goal};
}

int MDP::MSIRRT::PlannerConnect::get_planning_horizon_frame() const
{
    return this->planning_horizon_frame;
}

int MDP::MSIRRT::PlannerConnect::get_planning_start_frame() const
{
    return this->planning_start_frame;
}

double MDP::MSIRRT::PlannerConnect::get_maximum_joint_space_speed() const
{
    return this->vmax;
}

std::size_t MDP::MSIRRT::PlannerConnect::get_conditionally_rejected_samples() const
{
    return this->conditionally_rejected_samples;
}

std::size_t MDP::MSIRRT::PlannerConnect::get_conditionally_rejected_kinematic_samples() const
{
    return this->conditionally_rejected_kinematic_samples;
}

std::size_t MDP::MSIRRT::PlannerConnect::get_conditionally_rejected_temporal_samples() const
{
    return this->conditionally_rejected_temporal_samples;
}

std::size_t MDP::MSIRRT::PlannerConnect::get_conditionally_bypassed_samples() const
{
    return this->conditionally_bypassed_samples;
}

std::size_t MDP::MSIRRT::PlannerConnect::get_proposed_samples() const
{
    return this->proposed_samples;
}

std::size_t MDP::MSIRRT::PlannerConnect::get_evaluated_samples() const
{
    return this->evaluated_samples;
}

double MDP::MSIRRT::PlannerConnect::get_conditional_exploration_base_rate() const
{
    return this->conditional_exploration_rate;
}

double MDP::MSIRRT::PlannerConnect::get_conditional_exploration_effective_rate() const
{
    return this->conditional_feedback->effective_rate();
}

double MDP::MSIRRT::PlannerConnect::get_conditional_rejection_window_ratio() const
{
    return this->conditional_feedback->rejection_ratio();
}

std::size_t MDP::MSIRRT::PlannerConnect::get_conditional_rejection_feedback_triggers() const
{
    return this->conditional_feedback->trigger_count();
}

std::size_t MDP::MSIRRT::PlannerConnect::get_conditional_feedback_boosted_samples() const
{
    return this->conditional_feedback_boosted_samples;
}

bool MDP::MSIRRT::PlannerConnect::get_rejection_feedback_enabled() const
{
    return this->rejection_feedback_enabled;
}

void MDP::MSIRRT::PlannerConnect::reset_conditional_exploration_feedback_stage()
{
    this->conditional_feedback->reset_stage();
}
