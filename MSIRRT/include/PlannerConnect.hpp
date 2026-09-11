#pragma once

#include <vector>
#include "config_read_writer/config_read.hpp"
#include "CollisionManager/CollisionManager.hpp"
#include "Tree.hpp"
#include "TreeRepair.hpp"
#include "Vertex.hpp"
#include <random>
#include <chrono>
#include <memory>
namespace MDP::MSIRRT
{

    struct PlannerConnectTestAccess;

    struct ExecutionAdvanceReport
    {
        bool advanced = false;
        bool goal_already_reached = false;
        int execution_frame = 0;
        std::vector<double> configuration;
        MDP::MSIRRT::TreeRerootReport start_tree;
    };

    class PlannerConnect
    {
    public:
        PlannerConnect(MDP::ConfigReader::SceneTask scene_task_, int random_seed);
        ~PlannerConnect();                                               // destructor
        PlannerConnect(const PlannerConnect &other) = delete;            // copy constructor
        PlannerConnect(PlannerConnect &&other) = delete;                 // move constructor
        PlannerConnect &operator=(const PlannerConnect &other) = delete; // copy assignment
        PlannerConnect &operator=(PlannerConnect &&other) = delete;      // move assignment

        bool solve();
        bool has_valid_solution() const;
        std::vector<MDP::MSIRRT::Vertex*> get_final_path() const;
        bool check_path(std::vector<MDP::MSIRRT::Vertex*>& path);
        int get_start_tree_vertices_count() const;
        int get_goal_tree_vertices_count() const;
        std::size_t get_start_tree_active_vertices_count() const;
        std::size_t get_goal_tree_active_vertices_count() const;
        MDP::MSIRRT::BidirectionalRepairReport update_prediction(MDP::ConfigReader::SceneTask updated_scene_task);
        std::size_t get_prediction_version() const;
        int get_planning_horizon_frame() const;
        std::size_t get_conditionally_rejected_samples() const;
        std::size_t get_conditionally_rejected_kinematic_samples() const;
        std::size_t get_conditionally_rejected_temporal_samples() const;
        std::size_t get_conditionally_bypassed_samples() const;
        std::size_t get_proposed_samples() const;
        std::size_t get_evaluated_samples() const;
        double get_conditional_exploration_base_rate() const;
        double get_conditional_exploration_effective_rate() const;
        double get_conditional_rejection_window_ratio() const;
        std::size_t get_conditional_rejection_feedback_triggers() const;
        std::size_t get_conditional_feedback_boosted_samples() const;
        bool get_rejection_feedback_enabled() const;
        void reset_conditional_exploration_feedback_stage();
        std::vector<double> configuration_at_frame(double frame) const;
        ExecutionAdvanceReport advance_start_to_frame(int frame);
        int get_planning_start_frame() const;
        double get_maximum_joint_space_speed() const;
        void set_max_planning_time(double seconds);
        bool get_cooperative_deadline_checks_enabled() const;
        std::size_t get_cooperative_deadline_check_triggers() const;
        bool get_reachable_ellipsoid_sampling_enabled() const;
        std::size_t get_reachable_ellipsoid_samples() const;
        bool get_horizon_bounded_safe_interval_queries_enabled() const;

    private:
        friend struct PlannerConnectTestAccess;
        MDP::ConfigReader::SceneTask scene_task;
        std::unique_ptr<MDP::CollisionManager> collision_manager;
        bool check_planner_termination_condition() const;
        bool cooperative_search_budget_exhausted();
        std::vector<MDP::MSIRRT::FrameRange> prediction_change_windows(const MDP::ConfigReader::SceneTask &updated_scene_task) const;
        bool is_tree_edge_valid(const MDP::MSIRRT::Tree *tree, const MDP::MSIRRT::Vertex &parent, const MDP::MSIRRT::Vertex &child);
        std::size_t reconnect_invalidated_vertices(MDP::MSIRRT::Tree *tree, const std::vector<MDP::MSIRRT::Vertex *> &invalidated_vertices);
        void ensure_roots_after_prediction_update();
        std::vector<std::pair<int, int>> get_planning_safe_intervals(
            const std::vector<double> &robot_angles,
            double *elapsed_seconds = nullptr);
        std::vector<std::pair<int, int>> clip_safe_intervals_to_horizon(const std::vector<std::pair<int, int>> &safe_intervals) const;
        bool is_sample_temporally_reachable(const MDP::MSIRRT::Vertex::VertexCoordType &coords, const std::vector<std::pair<int, int>> &safe_intervals);
        bool is_sample_kinematically_reachable(const MDP::MSIRRT::Vertex::VertexCoordType &coords);
        std::pair<double, double> tree_conditioned_time_bounds(
            const MDP::MSIRRT::Vertex::VertexCoordType &coords);
        std::vector<MDP::MSIRRT::Vertex *> nearest_active_vertices(
            MDP::MSIRRT::Tree *tree,
            const MDP::MSIRRT::Vertex::VertexCoordType &coords,
            std::size_t count);

        bool is_coords_in_limits(const MDP::MSIRRT::Vertex &q) const;
        bool is_coords_in_limits(const MDP::MSIRRT::Vertex::VertexCoordType &coords) const;

        bool is_goal(const MDP::MSIRRT::Vertex::VertexCoordType &coord);
        double get_random_between_0_1();
        MDP::MSIRRT::Vertex::VertexCoordType get_random_configuration();
        MDP::MSIRRT::Vertex::VertexCoordType get_uniform_random_configuration();
        MDP::MSIRRT::Vertex::VertexCoordType get_reachable_ellipsoid_configuration();
        std::vector<MDP::MSIRRT::Vertex *> extend_naive(MDP::MSIRRT::Vertex::VertexCoordType &coords_of_new);

        MDP::MSIRRT::Vertex *get_nearest_node(const MDP::MSIRRT::Vertex::VertexCoordType &coords);
        std::vector<std::pair<MDP::MSIRRT::Vertex *, int>> get_nearest_node_by_radius(const MDP::MSIRRT::Vertex::VertexCoordType &coords, double raduis, MDP::MSIRRT::Tree *tree);

        bool extend(MDP::MSIRRT::Vertex::VertexCoordType &coords_of_new);
        std::vector<MDP::MSIRRT::Vertex *> set_parent(MDP::MSIRRT::Vertex::VertexCoordType &coord_rand, std::vector<std::pair<int, int>> &safe_intervals_of_coord_rand);
        std::vector<MDP::MSIRRT::Vertex *> rewire(MDP::MSIRRT::Vertex *node);
        std::pair<int, int> calculate_delta(MDP::MSIRRT::Vertex *candidate_node, MDP::MSIRRT::Vertex::VertexCoordType &end_coords, std::vector<std::pair<int, int>> &safe_intervals_of_coord_rand, int &safe_interval_ind);
        bool is_collision_motion(const MDP::MSIRRT::Vertex::VertexCoordType &start_coords, const MDP::MSIRRT::Vertex::VertexCoordType &end_coords, double &start_time, double &end_time);
        bool is_collision_motion(const MDP::MSIRRT::Vertex::VertexCoordType &start_coords, const MDP::MSIRRT::Vertex::VertexCoordType &end_coords, double &start_time, double &end_time, MDP::MSIRRT::Vertex::VertexCoordType& collision_coord);
        bool is_collision_state(MDP::MSIRRT::Vertex::VertexCoordType &coords, int &time);
        // bool is_collision_motion(const MDP::MSIRRT::Vertex::VertexCoordType& start_coords, const  MDP::MSIRRT::Vertex::VertexCoordType& end_coords, double& start_time, double& end_time,MDP::MSIRRT::Vertex::VertexCoordType& last_valid_coord, int& last_valid_time);
        std::vector<MDP::MSIRRT::Vertex*> grow_tree(MDP::MSIRRT::Vertex::VertexCoordType &coord_rand, std::vector<std::pair<int, int>> &safe_intervals_of_coord_rand);
        bool connect_trees(MDP::MSIRRT::Vertex::VertexCoordType& coord_rand, std::vector<std::pair<int, int>>& safe_intervals_of_coord_rand,std::vector<MDP::MSIRRT::Vertex* > new_nodes);
        void swap_trees();
        void prune_goal_tree();
        void warmup_start_tree();
        void warmup_goal_tree();
        MDP::MSIRRT::Tree *start_tree;
        MDP::MSIRRT::Tree *goal_tree;
        MDP::MSIRRT::Tree *orphan_tree;
        MDP::MSIRRT::Tree *current_tree;
        MDP::MSIRRT::Tree *other_tree;
        MDP::MSIRRT::Vertex *root_node = nullptr;

        MDP::MSIRRT::Vertex::VertexCoordType goal_coords;
        float max_planning_time = 180;
        bool stop_when_path_found = true;
        std::uniform_real_distribution<> probability_gen;
        std::uniform_real_distribution<> conditional_probability_gen;
        int dof = -1;
        double goal_bias = 0.4;
        double vmax = MDP::MSIRRT::DEFAULT_MAXIMUM_JOINT_SPACE_SPEED;
        double planner_range;
        double radius_factor = 4.0;
        // Hysteresis (in frames) before the static-obstacle "jump" path
        // triggers in set_parent. 0 = MSIRRT_jump baseline (jump on first
        // collision). 50 picked from a dense sweep across 1-280 obstacles
        // with 10 seeds/scene × 50 scenes/bucket: it minimises mean runtime
        // on 100/140/200/240 buckets, gives the best median at 200 (2.27s vs
        // 3.26 at 10) and the best combined success rate on 200/240/280
        // (83/52/22 %). On light scenes (20/40) it is within noise of the
        // optimum. Overridable via MSIRRT_MAX_DEP_TIME.
        int max_dep_time_for_safe_int_check = 0;
        bool goal_reached = false;
        std::pair<MDP::MSIRRT::Vertex *,MDP::MSIRRT::Vertex *> goal_nodes = std::pair<MDP::MSIRRT::Vertex *,MDP::MSIRRT::Vertex *>(nullptr,nullptr);
        MDP::MSIRRT::Vertex *finish_node = nullptr;
        std::vector<std::pair<float, float>> robot_limits;
        std::chrono::time_point<std::chrono::steady_clock> solver_start_time;
        std::random_device rd; // Will be used to obtain a seed for the random number engine
        std::mt19937 gen;
        std::mt19937 conditional_gen;
        bool goal_sampled;
        std::vector<std::pair<int, int>> goal_safe_intervals;
        std::size_t prediction_version = 0;
        int planning_horizon_frame = 0;
        int planning_start_frame = 0;
        bool executed_prefix = false;
        bool conditional_sampling_enabled = false;
        double conditional_exploration_rate = 0.1;
        double conditional_exploration_boost = 0.35;
        std::size_t conditional_rejection_window = 32;
        double conditional_rejection_threshold = 0.95;
        bool rejection_feedback_enabled = false;
        bool reuse_stored_previous_intervals = false;
        bool cooperative_deadline_checks_enabled = false;
        bool reachable_ellipsoid_sampling_enabled = false;
        bool horizon_bounded_safe_interval_queries_enabled = false;
        bool search_budget_active = false;
        std::size_t cooperative_deadline_check_triggers = 0;
        std::size_t reachable_ellipsoid_samples = 0;
        std::unique_ptr<MDP::MSIRRT::ConditionalExplorationFeedback> conditional_feedback;
        std::size_t conditional_feedback_boosted_samples = 0;
        std::size_t evaluated_samples = 0;
        std::size_t proposed_samples = 0;
        std::size_t conditionally_rejected_samples = 0;
        std::size_t conditionally_rejected_kinematic_samples = 0;
        std::size_t conditionally_rejected_temporal_samples = 0;
        std::size_t conditionally_bypassed_samples = 0;


        bool was_static_obstacle = false;
        MDP::MSIRRT::Vertex::VertexCoordType last_valid_coord;
    };
}
