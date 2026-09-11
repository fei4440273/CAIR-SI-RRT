#include "RepairTypes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace MDP::MSIRRT
{

bool FrameRange::contains(int frame) const
{
    return first <= frame && frame <= last;
}

bool FrameRange::overlaps(const FrameRange &other) const
{
    return first <= other.last && other.first <= last;
}

bool FrameRange::operator==(const FrameRange &other) const
{
    return first == other.first && last == other.last;
}

std::vector<FrameRange> normalize_frame_ranges(const std::vector<FrameRange> &ranges)
{
    std::vector<FrameRange> sorted = ranges;
    for (const auto &range : sorted)
    {
        if (range.first > range.last)
        {
            throw std::invalid_argument("frame range starts after it ends");
        }
    }

    std::sort(sorted.begin(), sorted.end(), [](const FrameRange &left, const FrameRange &right) {
        return left.first < right.first || (left.first == right.first && left.last < right.last);
    });

    std::vector<FrameRange> result;
    for (const auto &range : sorted)
    {
        if (result.empty() || range.first > result.back().last + 1)
        {
            result.push_back(range);
            continue;
        }
        result.back().last = std::max(result.back().last, range.last);
    }
    return result;
}

std::optional<FrameRange> find_range_containing(const std::vector<FrameRange> &ranges, int frame)
{
    for (const auto &range : ranges)
    {
        if (range.contains(frame))
        {
            return range;
        }
    }
    return std::nullopt;
}

IntervalTransition classify_interval_transition(
    const FrameRange &previous_interval,
    const std::vector<FrameRange> &current_intervals,
    double arrival_time)
{
    if (previous_interval.first > previous_interval.last || !std::isfinite(arrival_time))
    {
        throw std::invalid_argument("interval transition input is invalid");
    }
    const auto normalized = normalize_frame_ranges(current_intervals);
    const auto matching = std::find_if(
        normalized.begin(), normalized.end(), [&](const FrameRange &range) {
            return range.first <= arrival_time && arrival_time <= range.last;
        });
    if (matching == normalized.end())
    {
        return IntervalTransition::Deleted;
    }

    const auto overlap_count = static_cast<std::size_t>(std::count_if(
        normalized.begin(), normalized.end(), [&](const FrameRange &range) {
            return range.overlaps(previous_interval);
        }));
    if (overlap_count > 1)
    {
        return IntervalTransition::Split;
    }
    if (*matching == previous_interval)
    {
        return IntervalTransition::Unchanged;
    }
    if (previous_interval.first <= matching->first &&
        matching->last <= previous_interval.last)
    {
        return IntervalTransition::Shrunk;
    }
    if (matching->first <= previous_interval.first &&
        previous_interval.last <= matching->last)
    {
        return IntervalTransition::Expanded;
    }
    return IntervalTransition::ShiftedOrMerged;
}

IntervalSetTransition classify_interval_set_transition(
    const std::vector<FrameRange> &before,
    const std::vector<FrameRange> &after)
{
    const auto normalized_before = normalize_frame_ranges(before);
    const auto normalized_after = normalize_frame_ranges(after);
    if (normalized_before == normalized_after)
    {
        return IntervalSetTransition::Unchanged;
    }
    if (normalized_after.empty())
    {
        return IntervalSetTransition::Deleted;
    }

    const auto overlaps_at_least_two = [](const auto &source, const auto &target) {
        return std::any_of(source.begin(), source.end(), [&](const auto &component) {
            return std::count_if(
                       target.begin(), target.end(), [&](const auto &candidate) {
                           return component.overlaps(candidate);
                       }) >= 2;
        });
    };
    if (overlaps_at_least_two(normalized_before, normalized_after))
    {
        return IntervalSetTransition::Split;
    }
    if (overlaps_at_least_two(normalized_after, normalized_before))
    {
        return IntervalSetTransition::Merged;
    }

    const auto contains_set = [](const auto &outer, const auto &inner) {
        return std::all_of(inner.begin(), inner.end(), [&](const auto &component) {
            return std::any_of(outer.begin(), outer.end(), [&](const auto &container) {
                return container.first <= component.first &&
                    component.last <= container.last;
            });
        });
    };
    const bool before_contains_after = contains_set(normalized_before, normalized_after);
    const bool after_contains_before = contains_set(normalized_after, normalized_before);
    if (before_contains_after && !after_contains_before)
    {
        return IntervalSetTransition::Shrunk;
    }
    if (after_contains_before && !before_contains_after)
    {
        return IntervalSetTransition::Expanded;
    }
    return IntervalSetTransition::Shifted;
}

std::vector<FrameRange> prediction_change_windows(
    const std::vector<PredictionSample> &before,
    const std::vector<PredictionSample> &after,
    double tolerance)
{
    if (before.size() != after.size())
    {
        throw std::invalid_argument("prediction snapshots have different frame counts");
    }
    if (!std::isfinite(tolerance) || tolerance < 0.0)
    {
        throw std::invalid_argument("prediction comparison tolerance must be finite and nonnegative");
    }

    std::vector<FrameRange> changed_frames;
    for (std::size_t frame = 0; frame < before.size(); ++frame)
    {
        bool changed = std::abs(before[frame].tube_radius - after[frame].tube_radius) > tolerance;
        for (std::size_t axis = 0; axis < before[frame].center.size() && !changed; ++axis)
        {
            changed = std::abs(before[frame].center[axis] - after[frame].center[axis]) > tolerance;
        }
        if (changed)
        {
            changed_frames.emplace_back(static_cast<int>(frame), static_cast<int>(frame));
        }
    }
    return normalize_frame_ranges(changed_frames);
}

TemporalDependencyIndex::TemporalDependencyIndex(int bucket_width) : bucket_width_(bucket_width)
{
    if (bucket_width_ <= 0)
    {
        throw std::invalid_argument("temporal dependency bucket width must be positive");
    }
}

int TemporalDependencyIndex::bucket_for_frame(int frame) const
{
    if (frame < 0)
    {
        throw std::invalid_argument("dependency frame must be nonnegative");
    }
    return frame / bucket_width_;
}

void TemporalDependencyIndex::add_to_buckets(
    std::size_t id,
    const FrameRange &range,
    std::unordered_map<int, std::unordered_set<std::size_t>> &buckets)
{
    if (range.first > range.last)
    {
        throw std::invalid_argument("dependency frame range starts after it ends");
    }
    for (int bucket = bucket_for_frame(range.first); bucket <= bucket_for_frame(range.last); ++bucket)
    {
        buckets[bucket].insert(id);
    }
}

void TemporalDependencyIndex::remove_from_buckets(
    std::size_t id,
    const FrameRange &range,
    std::unordered_map<int, std::unordered_set<std::size_t>> &buckets)
{
    for (int bucket = bucket_for_frame(range.first); bucket <= bucket_for_frame(range.last); ++bucket)
    {
        const auto bucket_it = buckets.find(bucket);
        if (bucket_it == buckets.end())
        {
            continue;
        }
        bucket_it->second.erase(id);
        if (bucket_it->second.empty())
        {
            buckets.erase(bucket_it);
        }
    }
}

void TemporalDependencyIndex::add_vertex(std::size_t vertex_id, const FrameRange &range)
{
    remove_vertex(vertex_id);
    vertex_ranges_[vertex_id] = range;
    add_to_buckets(vertex_id, range, vertex_buckets_);
}

void TemporalDependencyIndex::add_edge(std::size_t edge_id, const FrameRange &range)
{
    remove_edge(edge_id);
    edge_ranges_[edge_id] = range;
    add_to_buckets(edge_id, range, edge_buckets_);
}

void TemporalDependencyIndex::remove_vertex(std::size_t vertex_id)
{
    const auto range_it = vertex_ranges_.find(vertex_id);
    if (range_it == vertex_ranges_.end())
    {
        return;
    }
    remove_from_buckets(vertex_id, range_it->second, vertex_buckets_);
    vertex_ranges_.erase(range_it);
}

void TemporalDependencyIndex::remove_edge(std::size_t edge_id)
{
    const auto range_it = edge_ranges_.find(edge_id);
    if (range_it == edge_ranges_.end())
    {
        return;
    }
    remove_from_buckets(edge_id, range_it->second, edge_buckets_);
    edge_ranges_.erase(range_it);
}

AffectedDependencies TemporalDependencyIndex::query(const std::vector<FrameRange> &changed_windows) const
{
    AffectedDependencies result;
    const auto normalized = normalize_frame_ranges(changed_windows);
    for (const auto &window : normalized)
    {
        for (int bucket = bucket_for_frame(window.first); bucket <= bucket_for_frame(window.last); ++bucket)
        {
            const auto vertex_bucket = vertex_buckets_.find(bucket);
            if (vertex_bucket != vertex_buckets_.end())
            {
                for (const auto id : vertex_bucket->second)
                {
                    if (vertex_ranges_.at(id).overlaps(window))
                    {
                        result.vertex_ids.insert(id);
                    }
                }
            }

            const auto edge_bucket = edge_buckets_.find(bucket);
            if (edge_bucket != edge_buckets_.end())
            {
                for (const auto id : edge_bucket->second)
                {
                    if (edge_ranges_.at(id).overlaps(window))
                    {
                        result.edge_ids.insert(id);
                    }
                }
            }
        }
    }
    return result;
}

void TemporalDependencyIndex::clear()
{
    vertex_ranges_.clear();
    edge_ranges_.clear();
    vertex_buckets_.clear();
    edge_buckets_.clear();
}

double RepairReport::vertex_reuse_rate() const
{
    if (active_vertices_before == 0)
    {
        return 1.0;
    }
    return static_cast<double>(reused_vertices) / static_cast<double>(active_vertices_before);
}

SearchContinuationBudget select_search_continuation_budget(
    double remaining_ms,
    double planning_call_guard_ms,
    double update_zero_final_guard_ms,
    bool update_zero,
    bool maximum_horizon,
    bool fallback_available)
{
    if (!std::isfinite(remaining_ms) || remaining_ms < 0.0 ||
        !std::isfinite(planning_call_guard_ms) || planning_call_guard_ms < 0.0 ||
        !std::isfinite(update_zero_final_guard_ms) ||
        update_zero_final_guard_ms < 5.0 || update_zero_final_guard_ms > 15.0)
    {
        throw std::invalid_argument("invalid search continuation budget input");
    }
    const bool special_case = update_zero && maximum_horizon && !fallback_available;
    const double effective_guard_ms = special_case
        ? update_zero_final_guard_ms
        : planning_call_guard_ms;
    const double solve_budget_ms = std::max(0.0, remaining_ms - effective_guard_ms);
    return {effective_guard_ms, solve_budget_ms, solve_budget_ms > 0.0};
}

double OnlineStageTiming::attributed_ms() const
{
    return prediction_ms + tube_update_ms + safe_interval_update_ms +
        tree_repair_ms + search_ms + fallback_ms;
}

double OnlineStageTiming::unattributed_ms(double wall_clock_ms) const
{
    return wall_clock_ms - attributed_ms();
}

bool OnlineStageTiming::reconciles(double wall_clock_ms, double tolerance_ms) const
{
    const std::array<double, 6> stages{
        prediction_ms,
        tube_update_ms,
        safe_interval_update_ms,
        tree_repair_ms,
        search_ms,
        fallback_ms,
    };
    if (!std::isfinite(wall_clock_ms) || wall_clock_ms < 0.0 ||
        !std::isfinite(tolerance_ms) || tolerance_ms < 0.0)
    {
        return false;
    }
    if (std::any_of(stages.begin(), stages.end(), [](double value) {
            return !std::isfinite(value) || value < 0.0;
        }))
    {
        return false;
    }
    return unattributed_ms(wall_clock_ms) >= -tolerance_ms;
}

PrimaryFailureCategory classify_primary_failure(const FailureAttributionInput &input)
{
    if (input.task_completed)
    {
        return PrimaryFailureCategory::None;
    }
    if (!input.result_valid)
    {
        return PrimaryFailureCategory::InvalidOrMissingResult;
    }
    if (input.safety_validation_failed)
    {
        return PrimaryFailureCategory::SafetyValidationFailed;
    }
    if (input.deadline_missed)
    {
        return PrimaryFailureCategory::DeadlineMissed;
    }
    if (input.prediction_unavailable)
    {
        return PrimaryFailureCategory::PredictionUnavailableStop;
    }
    if (input.reliable_horizon_too_short)
    {
        return PrimaryFailureCategory::ReliableHorizonTooShort;
    }
    if (!input.ever_accepted_plan)
    {
        return PrimaryFailureCategory::InitialPlanningFailed;
    }
    return PrimaryFailureCategory::PostChangeRepairFailed;
}

const char *primary_failure_category_name(PrimaryFailureCategory category)
{
    switch (category)
    {
    case PrimaryFailureCategory::None:
        return "none";
    case PrimaryFailureCategory::InitialPlanningFailed:
        return "initial_planning_failed";
    case PrimaryFailureCategory::PostChangeRepairFailed:
        return "post_change_repair_failed";
    case PrimaryFailureCategory::ReliableHorizonTooShort:
        return "reliable_horizon_too_short";
    case PrimaryFailureCategory::DeadlineMissed:
        return "deadline_missed";
    case PrimaryFailureCategory::PredictionUnavailableStop:
        return "prediction_unavailable_stop";
    case PrimaryFailureCategory::SafetyValidationFailed:
        return "safety_validation_failed";
    case PrimaryFailureCategory::InvalidOrMissingResult:
        return "invalid_or_missing_result";
    }
    throw std::invalid_argument("unknown primary failure category");
}

void accumulate_sequential_repair_report(RepairReport &target, const RepairReport &addition)
{
    if (!target.reuse_identity_initialized)
    {
        target.active_vertices_before = addition.active_vertices_before;
        target.reused_vertex_ids = addition.reused_vertex_ids;
        target.reuse_identity_initialized = addition.reuse_identity_initialized;
        target.reused_vertices = addition.reused_vertices;
    }
    else if (addition.reuse_identity_initialized)
    {
        for (auto vertex = target.reused_vertex_ids.begin(); vertex != target.reused_vertex_ids.end();)
        {
            if (addition.reused_vertex_ids.count(*vertex) == 0)
            {
                vertex = target.reused_vertex_ids.erase(vertex);
            }
            else
            {
                ++vertex;
            }
        }
        target.reused_vertices = target.reused_vertex_ids.size();
    }
    else
    {
        target.reused_vertices = std::min({
            target.reused_vertices,
            addition.reused_vertices,
            target.active_vertices_before});
    }

    target.candidate_vertices += addition.candidate_vertices;
    target.candidate_edges += addition.candidate_edges;
    target.revalidated_vertices += addition.revalidated_vertices;
    target.revalidated_edges += addition.revalidated_edges;
    target.invalidated_vertices += addition.invalidated_vertices;
    target.reconnected_vertices += addition.reconnected_vertices;
    target.intervals_unchanged += addition.intervals_unchanged;
    target.intervals_shrunk += addition.intervals_shrunk;
    target.intervals_expanded += addition.intervals_expanded;
    target.intervals_split += addition.intervals_split;
    target.intervals_shifted += addition.intervals_shifted;
    target.intervals_merged += addition.intervals_merged;
    target.intervals_deleted += addition.intervals_deleted;
    target.invariants_hold = target.invariants_hold && addition.invariants_hold;
    if (!addition.invariant_error.empty())
    {
        target.invariant_error = addition.invariant_error;
    }
}

std::vector<MotionValidationSample> motion_validation_samples(
    double start_time,
    double end_time,
    int interpolation_steps)
{
    if (!std::isfinite(start_time) || !std::isfinite(end_time) || start_time >= end_time)
    {
        throw std::invalid_argument("motion validation time range is invalid");
    }
    if (interpolation_steps <= 0)
    {
        throw std::invalid_argument("motion validation interpolation steps must be positive");
    }

    std::vector<MotionValidationSample> samples;
    const double duration = end_time - start_time;
    const auto add_sample = [&](double alpha, int frame) {
        const auto duplicate = std::find_if(samples.begin(), samples.end(), [&](const auto &sample) {
            return sample.frame == frame && std::abs(sample.alpha - alpha) <= 1e-12;
        });
        if (duplicate == samples.end())
        {
            samples.push_back({alpha, frame});
        }
    };

    for (int step = 0; step <= interpolation_steps; ++step)
    {
        const double alpha = static_cast<double>(step) / interpolation_steps;
        add_sample(alpha, static_cast<int>(std::floor(start_time + duration * alpha)));
    }

    const int first_integer_frame = static_cast<int>(std::ceil(start_time));
    const int last_integer_frame = static_cast<int>(std::floor(end_time));
    for (int frame = first_integer_frame; frame <= last_integer_frame; ++frame)
    {
        add_sample((frame - start_time) / duration, frame);
    }
    const int arrival_frame = static_cast<int>(std::ceil(end_time));
    if (arrival_frame > last_integer_frame)
    {
        add_sample(1.0, arrival_frame);
    }

    std::stable_sort(samples.begin(), samples.end(), [](const auto &left, const auto &right) {
        if (left.alpha != right.alpha)
        {
            return left.alpha < right.alpha;
        }
        return left.frame < right.frame;
    });
    return samples;
}

ExecutionWindow select_execution_window(
    int current_frame,
    int next_issue_frame,
    int reliable_until_frame,
    int path_until_frame)
{
    if (current_frame < 0 || next_issue_frame < current_frame)
    {
        throw std::invalid_argument("execution frame cannot move backwards");
    }
    if (reliable_until_frame < current_frame)
    {
        throw std::invalid_argument("prediction reliability is exhausted before execution starts");
    }
    if (path_until_frame < current_frame)
    {
        throw std::invalid_argument("validated path ends before execution starts");
    }

    ExecutionWindow window;
    window.requested_frame = next_issue_frame;
    window.reliable_until_frame = reliable_until_frame;
    window.path_until_frame = path_until_frame;
    window.validated_frame = std::min({
        next_issue_frame,
        reliable_until_frame,
        path_until_frame});
    window.prefix_frames = window.validated_frame - current_frame;
    window.shortened_frames = next_issue_frame - window.validated_frame;
    if (window.prefix_frames == 0)
    {
        window.limit_reason = "no_executable_prefix";
    }
    else if (window.validated_frame == next_issue_frame)
    {
        window.limit_reason = "next_prediction";
    }
    else if (reliable_until_frame <= path_until_frame)
    {
        window.limit_reason = "prediction_reliability";
    }
    else
    {
        window.limit_reason = "validated_path";
    }
    window.limited_by_reliability =
        window.limit_reason == "prediction_reliability";
    return window;
}

PlanningBudget allocate_planning_budget(
    double deadline_ms,
    double update_ms,
    double reserve_fraction,
    double reserve_min_ms)
{
    if (!std::isfinite(deadline_ms) || deadline_ms <= 0.0 ||
        !std::isfinite(update_ms) || update_ms < 0.0 ||
        !std::isfinite(reserve_fraction) || reserve_fraction < 0.0 ||
        reserve_fraction >= 1.0 || !std::isfinite(reserve_min_ms) ||
        reserve_min_ms < 0.0)
    {
        throw std::invalid_argument("planning budget inputs are invalid");
    }
    PlanningBudget result;
    result.available_after_update_ms = std::max(0.0, deadline_ms - update_ms);
    result.fallback_reserved_ms = std::min(
        result.available_after_update_ms,
        std::max(
            reserve_min_ms,
            reserve_fraction * result.available_after_update_ms));
    result.progressive_ms =
        result.available_after_update_ms - result.fallback_reserved_ms;
    return result;
}

double initial_stage_budget_ms(
    int base_stage_ms,
    int skipped_stages,
    double reuse_fraction,
    double progressive_available_ms,
    double guard_ms)
{
    if (base_stage_ms <= 0 || skipped_stages < 0 ||
        !std::isfinite(reuse_fraction) || reuse_fraction < 0.0 ||
        reuse_fraction > 1.0 || !std::isfinite(progressive_available_ms) ||
        progressive_available_ms < 0.0 || !std::isfinite(guard_ms) ||
        guard_ms < 0.0)
    {
        throw std::invalid_argument("initial stage budget inputs are invalid");
    }
    const double requested = static_cast<double>(base_stage_ms) *
        (1.0 + static_cast<double>(skipped_stages) * reuse_fraction);
    return std::min(
        requested,
        std::max(0.0, progressive_available_ms - guard_ms));
}

double fallback_solve_budget_ms(
    double remaining_ms,
    double setup_estimate_ms,
    double deadline_guard_ms)
{
    if (!std::isfinite(remaining_ms) || remaining_ms < 0.0 ||
        !std::isfinite(setup_estimate_ms) || setup_estimate_ms < 0.0 ||
        !std::isfinite(deadline_guard_ms) || deadline_guard_ms < 0.0)
    {
        throw std::invalid_argument("fallback budget inputs are invalid");
    }
    return std::max(
        0.0, remaining_ms - setup_estimate_ms - deadline_guard_ms);
}

int minimum_kinematic_arrival_frame(
    const std::vector<double> &start_configuration,
    const std::vector<double> &goal_configuration,
    int start_frame,
    int fps,
    double maximum_joint_space_speed)
{
    if (start_configuration.size() != goal_configuration.size() ||
        start_configuration.empty() || start_frame < 0 || fps <= 0 ||
        !std::isfinite(maximum_joint_space_speed) ||
        maximum_joint_space_speed <= 0.0)
    {
        throw std::invalid_argument("kinematic arrival inputs are invalid");
    }
    double squared_distance = 0.0;
    for (std::size_t joint = 0; joint < start_configuration.size(); ++joint)
    {
        if (!std::isfinite(start_configuration[joint]) ||
            !std::isfinite(goal_configuration[joint]))
        {
            throw std::invalid_argument("kinematic arrival configurations must be finite");
        }
        const double difference = goal_configuration[joint] - start_configuration[joint];
        squared_distance += difference * difference;
    }
    const double travel_frames =
        std::sqrt(squared_distance) * static_cast<double>(fps) /
        maximum_joint_space_speed;
    return static_cast<int>(std::ceil(static_cast<double>(start_frame) + travel_frames));
}

InitialHorizonSelection select_initial_planning_horizon(
    int issue_frame,
    int configured_initial_frames,
    int minimum_required_frame,
    int maximum_reliable_frame,
    int feasibility_slack_frames,
    int horizon_step_frames,
    bool feasibility_aware_enabled)
{
    if (issue_frame < 0 || configured_initial_frames < 0 ||
        minimum_required_frame < issue_frame ||
        maximum_reliable_frame < issue_frame || feasibility_slack_frames < 0 ||
        horizon_step_frames <= 0)
    {
        throw std::invalid_argument("initial planning horizon inputs are invalid");
    }

    InitialHorizonSelection selection;
    selection.raw_frame = std::min(
        maximum_reliable_frame, issue_frame + configured_initial_frames);
    selection.selected_frame = selection.raw_frame;
    if (feasibility_aware_enabled)
    {
        const int feasibility_target = std::min(
            maximum_reliable_frame,
            std::max(
                selection.raw_frame,
                minimum_required_frame + feasibility_slack_frames));
        const int stages_to_skip =
            (feasibility_target - selection.raw_frame + horizon_step_frames - 1) /
            horizon_step_frames;
        selection.selected_frame = std::min(
            maximum_reliable_frame,
            selection.raw_frame + stages_to_skip * horizon_step_frames);
        selection.skipped_infeasible_stages = stages_to_skip;
    }
    selection.adjusted_for_feasibility =
        selection.selected_frame > selection.raw_frame;
    return selection;
}

ConditionalExplorationFeedback::ConditionalExplorationFeedback(
    double base_rate,
    double boost,
    std::size_t window_size,
    double rejection_threshold,
    bool enabled)
    : base_rate_(base_rate),
      boost_(boost),
      window_size_(window_size),
      rejection_threshold_(rejection_threshold),
      enabled_(enabled)
{
    if (!std::isfinite(base_rate_) || base_rate_ < 0.0 || base_rate_ > 1.0 ||
        !std::isfinite(boost_) || boost_ < 0.0 || boost_ > 1.0 ||
        window_size_ == 0 || !std::isfinite(rejection_threshold_) ||
        rejection_threshold_ < 0.0 || rejection_threshold_ > 1.0)
    {
        throw std::invalid_argument("conditional exploration feedback inputs are invalid");
    }
}

void ConditionalExplorationFeedback::observe(bool rejected)
{
    if (!enabled_)
    {
        return;
    }
    const bool was_active = active_;
    outcomes_.push_back(rejected);
    rejection_count_ += rejected ? 1U : 0U;
    if (outcomes_.size() > window_size_)
    {
        rejection_count_ -= outcomes_.front() ? 1U : 0U;
        outcomes_.pop_front();
    }
    active_ = outcomes_.size() == window_size_ &&
        rejection_ratio() >= rejection_threshold_;
    if (!was_active && active_)
    {
        ++trigger_count_;
    }
}

void ConditionalExplorationFeedback::reset_stage()
{
    outcomes_.clear();
    rejection_count_ = 0;
    active_ = false;
}

double ConditionalExplorationFeedback::effective_rate() const
{
    return active_ ? std::min(1.0, base_rate_ + boost_) : base_rate_;
}

double ConditionalExplorationFeedback::rejection_ratio() const
{
    if (outcomes_.empty())
    {
        return 0.0;
    }
    return static_cast<double>(rejection_count_) /
        static_cast<double>(outcomes_.size());
}

std::size_t ConditionalExplorationFeedback::trigger_count() const
{
    return trigger_count_;
}

bool ConditionalExplorationFeedback::active() const
{
    return active_;
}

} // namespace MDP::MSIRRT
