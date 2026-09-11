#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace MDP::MSIRRT
{

inline constexpr double DEFAULT_MAXIMUM_JOINT_SPACE_SPEED = 3.1415;

struct FrameRange
{
    int first = 0;
    int last = -1;

    FrameRange() = default;
    FrameRange(int first_, int last_) : first(first_), last(last_) {}

    bool contains(int frame) const;
    bool overlaps(const FrameRange &other) const;
    bool operator==(const FrameRange &other) const;
};

std::vector<FrameRange> normalize_frame_ranges(const std::vector<FrameRange> &ranges);
std::optional<FrameRange> find_range_containing(const std::vector<FrameRange> &ranges, int frame);

enum class IntervalTransition
{
    Unchanged,
    Shrunk,
    Expanded,
    Split,
    ShiftedOrMerged,
    Deleted,
};

IntervalTransition classify_interval_transition(
    const FrameRange &previous_interval,
    const std::vector<FrameRange> &current_intervals,
    double arrival_time);

enum class IntervalSetTransition
{
    Unchanged,
    Shrunk,
    Expanded,
    Split,
    Merged,
    Shifted,
    Deleted,
};

IntervalSetTransition classify_interval_set_transition(
    const std::vector<FrameRange> &before,
    const std::vector<FrameRange> &after);

struct PredictionSample
{
    std::array<double, 3> center{};
    double tube_radius = 0.0;
};

std::vector<FrameRange> prediction_change_windows(
    const std::vector<PredictionSample> &before,
    const std::vector<PredictionSample> &after,
    double tolerance);

struct AffectedDependencies
{
    std::unordered_set<std::size_t> vertex_ids;
    std::unordered_set<std::size_t> edge_ids;
};

class TemporalDependencyIndex
{
public:
    explicit TemporalDependencyIndex(int bucket_width = 16);

    void add_vertex(std::size_t vertex_id, const FrameRange &range);
    void add_edge(std::size_t edge_id, const FrameRange &range);
    void remove_vertex(std::size_t vertex_id);
    void remove_edge(std::size_t edge_id);
    AffectedDependencies query(const std::vector<FrameRange> &changed_windows) const;
    void clear();

private:
    int bucket_for_frame(int frame) const;
    void add_to_buckets(
        std::size_t id,
        const FrameRange &range,
        std::unordered_map<int, std::unordered_set<std::size_t>> &buckets);
    void remove_from_buckets(
        std::size_t id,
        const FrameRange &range,
        std::unordered_map<int, std::unordered_set<std::size_t>> &buckets);

    int bucket_width_;
    std::unordered_map<std::size_t, FrameRange> vertex_ranges_;
    std::unordered_map<std::size_t, FrameRange> edge_ranges_;
    std::unordered_map<int, std::unordered_set<std::size_t>> vertex_buckets_;
    std::unordered_map<int, std::unordered_set<std::size_t>> edge_buckets_;
};

struct RepairReport
{
    std::size_t active_vertices_before = 0;
    std::size_t candidate_vertices = 0;
    std::size_t candidate_edges = 0;
    std::size_t revalidated_vertices = 0;
    std::size_t revalidated_edges = 0;
    std::size_t reused_vertices = 0;
    std::size_t invalidated_vertices = 0;
    std::size_t reconnected_vertices = 0;
    std::size_t intervals_unchanged = 0;
    std::size_t intervals_shrunk = 0;
    std::size_t intervals_expanded = 0;
    std::size_t intervals_split = 0;
    std::size_t intervals_shifted = 0;
    std::size_t intervals_merged = 0;
    std::size_t intervals_deleted = 0;
    std::unordered_set<std::size_t> reused_vertex_ids;
    bool reuse_identity_initialized = false;
    bool invariants_hold = false;
    std::string invariant_error;

    double vertex_reuse_rate() const;
};

void accumulate_sequential_repair_report(RepairReport &target, const RepairReport &addition);

struct MotionValidationSample
{
    double alpha = 0.0;
    int frame = 0;
};

std::vector<MotionValidationSample> motion_validation_samples(
    double start_time,
    double end_time,
    int interpolation_steps);

struct ExecutionWindow
{
    int requested_frame = 0;
    int reliable_until_frame = 0;
    int path_until_frame = 0;
    int validated_frame = 0;
    int prefix_frames = 0;
    int shortened_frames = 0;
    bool limited_by_reliability = false;
    std::string limit_reason = "no_executable_prefix";
};

ExecutionWindow select_execution_window(
    int current_frame,
    int next_issue_frame,
    int reliable_until_frame,
    int path_until_frame);

struct PlanningBudget
{
    double available_after_update_ms = 0.0;
    double progressive_ms = 0.0;
    double fallback_reserved_ms = 0.0;
};

struct SearchContinuationBudget
{
    double effective_guard_ms = 0.0;
    double solve_budget_ms = 0.0;
    bool allowed = false;
};

SearchContinuationBudget select_search_continuation_budget(
    double remaining_ms,
    double planning_call_guard_ms,
    double update_zero_final_guard_ms,
    bool update_zero,
    bool maximum_horizon,
    bool fallback_available);

struct OnlineStageTiming
{
    double prediction_ms = 0.0;
    double tube_update_ms = 0.0;
    double safe_interval_update_ms = 0.0;
    double tree_repair_ms = 0.0;
    double search_ms = 0.0;
    double fallback_ms = 0.0;

    double attributed_ms() const;
    double unattributed_ms(double wall_clock_ms) const;
    bool reconciles(double wall_clock_ms, double tolerance_ms = 0.1) const;
};

enum class PrimaryFailureCategory
{
    None,
    InitialPlanningFailed,
    PostChangeRepairFailed,
    ReliableHorizonTooShort,
    DeadlineMissed,
    PredictionUnavailableStop,
    SafetyValidationFailed,
    InvalidOrMissingResult,
};

struct FailureAttributionInput
{
    bool task_completed = false;
    bool result_valid = true;
    bool safety_validation_failed = false;
    bool deadline_missed = false;
    bool prediction_unavailable = false;
    bool reliable_horizon_too_short = false;
    bool ever_accepted_plan = false;
    std::size_t terminal_update_index = 0;
};

PrimaryFailureCategory classify_primary_failure(const FailureAttributionInput &input);
const char *primary_failure_category_name(PrimaryFailureCategory category);

PlanningBudget allocate_planning_budget(
    double deadline_ms,
    double update_ms,
    double reserve_fraction,
    double reserve_min_ms);

double initial_stage_budget_ms(
    int base_stage_ms,
    int skipped_stages,
    double reuse_fraction,
    double progressive_available_ms,
    double guard_ms);

double fallback_solve_budget_ms(
    double remaining_ms,
    double setup_estimate_ms,
    double deadline_guard_ms);

int minimum_kinematic_arrival_frame(
    const std::vector<double> &start_configuration,
    const std::vector<double> &goal_configuration,
    int start_frame,
    int fps,
    double maximum_joint_space_speed);

struct InitialHorizonSelection
{
    int raw_frame = 0;
    int selected_frame = 0;
    int skipped_infeasible_stages = 0;
    bool adjusted_for_feasibility = false;
};

InitialHorizonSelection select_initial_planning_horizon(
    int issue_frame,
    int configured_initial_frames,
    int minimum_required_frame,
    int maximum_reliable_frame,
    int feasibility_slack_frames,
    int horizon_step_frames,
    bool feasibility_aware_enabled);

class ConditionalExplorationFeedback
{
public:
    ConditionalExplorationFeedback(
        double base_rate,
        double boost,
        std::size_t window_size,
        double rejection_threshold,
        bool enabled);

    void observe(bool rejected);
    void reset_stage();
    double effective_rate() const;
    double rejection_ratio() const;
    std::size_t trigger_count() const;
    bool active() const;

private:
    double base_rate_;
    double boost_;
    std::size_t window_size_;
    double rejection_threshold_;
    bool enabled_;
    std::deque<bool> outcomes_;
    std::size_t rejection_count_ = 0;
    std::size_t trigger_count_ = 0;
    bool active_ = false;
};

struct BidirectionalRepairReport
{
    std::size_t prediction_version = 0;
    std::vector<FrameRange> changed_windows;
    RepairReport start_tree;
    RepairReport goal_tree;
    double tube_update_seconds = 0.0;
    double safe_interval_update_seconds = 0.0;
    double tree_repair_seconds = 0.0;
    double collision_structure_update_seconds = 0.0;
    std::size_t collision_frames_rebuilt = 0;
    std::size_t collision_obstacles_reindexed = 0;
    bool collision_full_rebuild_fallback = false;
    double repair_seconds = 0.0;
    bool used_stored_previous_intervals = false;
    bool previous_solution_invalidated = false;
    bool previous_solution_preserved = false;
    bool invariants_hold = false;
};

} // namespace MDP::MSIRRT
