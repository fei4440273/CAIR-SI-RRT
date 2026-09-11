#include "RepairTypes.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

using MDP::MSIRRT::FrameRange;
using MDP::MSIRRT::IntervalTransition;
using MDP::MSIRRT::IntervalSetTransition;
using MDP::MSIRRT::PredictionSample;
using MDP::MSIRRT::TemporalDependencyIndex;

namespace {

void test_normalize_and_select_intervals()
{
    const std::vector<FrameRange> input{{5, 7}, {1, 2}, {3, 4}, {11, 12}, {10, 11}};
    const auto normalized = MDP::MSIRRT::normalize_frame_ranges(input);

    assert(normalized.size() == 2);
    assert(normalized[0] == FrameRange(1, 7));
    assert(normalized[1] == FrameRange(10, 12));
    assert(MDP::MSIRRT::find_range_containing(normalized, 6).value() == FrameRange(1, 7));
    assert(!MDP::MSIRRT::find_range_containing(normalized, 9).has_value());
}

void test_prediction_change_windows()
{
    const std::vector<PredictionSample> before{
        {{0.0, 0.0, 0.0}, 0.01},
        {{0.1, 0.0, 0.0}, 0.02},
        {{0.2, 0.0, 0.0}, 0.03},
        {{0.3, 0.0, 0.0}, 0.04},
        {{0.4, 0.0, 0.0}, 0.05},
    };
    auto after = before;
    after[1].center[0] += 0.2;
    after[2].tube_radius += 0.03;
    after[4].center[1] += 0.2;

    const auto changed = MDP::MSIRRT::prediction_change_windows(before, after, 1e-6);
    assert(changed.size() == 2);
    assert(changed[0] == FrameRange(1, 2));
    assert(changed[1] == FrameRange(4, 4));
}

void test_interval_transition_classification()
{
    using MDP::MSIRRT::classify_interval_transition;
    assert(classify_interval_transition({0, 10}, {{0, 10}}, 5.0) ==
           IntervalTransition::Unchanged);
    assert(classify_interval_transition({0, 10}, {{2, 8}}, 5.0) ==
           IntervalTransition::Shrunk);
    assert(classify_interval_transition({2, 8}, {{0, 10}}, 5.0) ==
           IntervalTransition::Expanded);
    assert(classify_interval_transition({0, 10}, {{0, 4}, {6, 10}}, 2.0) ==
           IntervalTransition::Split);
    assert(classify_interval_transition({2, 8}, {{4, 10}}, 5.0) ==
           IntervalTransition::ShiftedOrMerged);
    assert(classify_interval_transition({0, 10}, {{6, 10}}, 5.0) ==
           IntervalTransition::Deleted);
}

void test_interval_set_transition_classification()
{
    using MDP::MSIRRT::classify_interval_set_transition;
    assert(classify_interval_set_transition({{0, 30}}, {{0, 5}, {10, 30}}) ==
           IntervalSetTransition::Split);
    assert(classify_interval_set_transition({{0, 5}, {10, 30}}, {{0, 30}}) ==
           IntervalSetTransition::Merged);
    assert(classify_interval_set_transition({{0, 10}}, {{5, 15}}) ==
           IntervalSetTransition::Shifted);
    assert(classify_interval_set_transition({{0, 10}}, {}) ==
           IntervalSetTransition::Deleted);
    assert(classify_interval_set_transition({{0, 10}}, {{2, 8}}) ==
           IntervalSetTransition::Shrunk);
    assert(classify_interval_set_transition({{2, 8}}, {{0, 10}}) ==
           IntervalSetTransition::Expanded);
    assert(classify_interval_set_transition({{0, 10}}, {{0, 10}}) ==
           IntervalSetTransition::Unchanged);
}

void test_temporal_dependency_index()
{
    TemporalDependencyIndex index;
    index.add_vertex(10, FrameRange(2, 4));
    index.add_vertex(11, FrameRange(8, 9));
    index.add_edge(20, FrameRange(5, 8));

    auto affected = index.query({FrameRange(4, 5)});
    assert(affected.vertex_ids.size() == 1);
    assert(affected.vertex_ids.count(10) == 1);
    assert(affected.edge_ids.size() == 1);
    assert(affected.edge_ids.count(20) == 1);

    index.remove_vertex(10);
    affected = index.query({FrameRange(0, 6)});
    assert(affected.vertex_ids.empty());
    assert(affected.edge_ids.count(20) == 1);

    index.clear();
    affected = index.query({FrameRange(0, 100)});
    assert(affected.vertex_ids.empty());
    assert(affected.edge_ids.empty());
}

void test_reuse_rate()
{
    MDP::MSIRRT::RepairReport report;
    report.active_vertices_before = 10;
    report.reused_vertices = 7;
    assert(std::abs(report.vertex_reuse_rate() - 0.7) < 1e-12);

    report.active_vertices_before = 0;
    assert(report.vertex_reuse_rate() == 1.0);
}

void test_sequential_repair_keeps_only_original_survivors()
{
    MDP::MSIRRT::RepairReport accumulated;
    MDP::MSIRRT::RepairReport first;
    first.active_vertices_before = 10;
    first.reused_vertex_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    first.reused_vertices = first.reused_vertex_ids.size();
    first.reuse_identity_initialized = true;
    first.invariants_hold = true;
    MDP::MSIRRT::accumulate_sequential_repair_report(accumulated, first);

    MDP::MSIRRT::RepairReport second;
    second.active_vertices_before = 12;
    second.reused_vertex_ids = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    second.reused_vertices = second.reused_vertex_ids.size();
    second.reuse_identity_initialized = true;
    second.invariants_hold = true;
    second.intervals_unchanged = 2;
    second.intervals_shrunk = 3;
    second.intervals_expanded = 4;
    second.intervals_split = 5;
    second.intervals_shifted = 6;
    second.intervals_merged = 7;
    second.intervals_deleted = 8;
    MDP::MSIRRT::accumulate_sequential_repair_report(accumulated, second);

    assert(accumulated.active_vertices_before == 10);
    assert(accumulated.reused_vertices == 6);
    assert(std::abs(accumulated.vertex_reuse_rate() - 0.6) < 1e-12);
    assert(accumulated.intervals_unchanged == 2);
    assert(accumulated.intervals_shrunk == 3);
    assert(accumulated.intervals_expanded == 4);
    assert(accumulated.intervals_split == 5);
    assert(accumulated.intervals_shifted == 6);
    assert(accumulated.intervals_merged == 7);
    assert(accumulated.intervals_deleted == 8);
}

void test_motion_validation_includes_exact_integer_frame_configuration()
{
    const double start = 36.30225006379864;
    const double end = 52.5308769348098;
    const auto samples = MDP::MSIRRT::motion_validation_samples(start, end, 17);

    const double expected_alpha = (43.0 - start) / (end - start);
    const auto exact_frame = std::find_if(samples.begin(), samples.end(), [&](const auto &sample) {
        return sample.frame == 43 && std::abs(sample.alpha - expected_alpha) < 1e-12;
    });
    assert(exact_frame != samples.end());

    const auto arrival_frame = std::find_if(samples.begin(), samples.end(), [&](const auto &sample) {
        return sample.frame == 53 && std::abs(sample.alpha - 1.0) < 1e-12;
    });
    assert(arrival_frame != samples.end());
}

void test_execution_window_is_bounded_by_prediction_reliability_and_path()
{
    const auto complete = MDP::MSIRRT::select_execution_window(10, 25, 50, 40);
    assert(complete.validated_frame == 25);
    assert(complete.limit_reason == "next_prediction");
    assert(complete.shortened_frames == 0);

    const auto reliable = MDP::MSIRRT::select_execution_window(10, 25, 18, 40);
    assert(reliable.validated_frame == 18);
    assert(reliable.prefix_frames == 8);
    assert(reliable.shortened_frames == 7);
    assert(reliable.limited_by_reliability);
    assert(reliable.limit_reason == "prediction_reliability");

    const auto path = MDP::MSIRRT::select_execution_window(10, 25, 50, 16);
    assert(path.validated_frame == 16);
    assert(path.path_until_frame == 16);
    assert(path.shortened_frames == 9);
    assert(path.limit_reason == "validated_path");

    const auto zero = MDP::MSIRRT::select_execution_window(10, 25, 10, 40);
    assert(zero.prefix_frames == 0);
    assert(zero.limit_reason == "no_executable_prefix");

    bool backwards_rejected = false;
    try
    {
        static_cast<void>(MDP::MSIRRT::select_execution_window(15, 14, 45, 40));
    }
    catch (const std::invalid_argument &)
    {
        backwards_rejected = true;
    }
    assert(backwards_rejected);

    bool exhausted_reliability_rejected = false;
    try
    {
        static_cast<void>(MDP::MSIRRT::select_execution_window(15, 30, 14, 40));
    }
    catch (const std::invalid_argument &)
    {
        exhausted_reliability_rejected = true;
    }
    assert(exhausted_reliability_rejected);

    bool exhausted_path_rejected = false;
    try
    {
        static_cast<void>(MDP::MSIRRT::select_execution_window(15, 30, 45, 14));
    }
    catch (const std::invalid_argument &)
    {
        exhausted_path_rejected = true;
    }
    assert(exhausted_path_rejected);
}

void test_planning_budget_reserves_full_replan_time()
{
    const auto budget = MDP::MSIRRT::allocate_planning_budget(100.0, 20.0, 0.25, 10.0);
    assert(std::abs(budget.progressive_ms - 60.0) < 1e-12);
    assert(std::abs(budget.fallback_reserved_ms - 20.0) < 1e-12);
    assert(std::abs(budget.available_after_update_ms - 80.0) < 1e-12);

    const auto exhausted = MDP::MSIRRT::allocate_planning_budget(100.0, 120.0, 0.25, 10.0);
    assert(exhausted.progressive_ms == 0.0);
    assert(exhausted.fallback_reserved_ms == 0.0);
}

void test_initial_stage_budget_reuses_skipped_budget_with_guard()
{
    using MDP::MSIRRT::initial_stage_budget_ms;
    assert(std::abs(initial_stage_budget_ms(15, 0, 1.0, 40.0, 15.0) - 15.0) < 1e-12);
    assert(std::abs(initial_stage_budget_ms(15, 1, 0.5, 40.0, 15.0) - 22.5) < 1e-12);
    assert(std::abs(initial_stage_budget_ms(15, 2, 1.0, 35.0, 15.0) - 20.0) < 1e-12);
    assert(std::abs(initial_stage_budget_ms(15, 1, 0.0, 40.0, 15.0) - 15.0) < 1e-12);

    for (const auto invalid : std::vector<std::array<double, 5>>{
             {0.0, 1.0, 0.5, 40.0, 15.0},
             {15.0, -1.0, 0.5, 40.0, 15.0},
             {15.0, 1.0, -0.1, 40.0, 15.0},
             {15.0, 1.0, 1.1, 40.0, 15.0},
             {15.0, 1.0, 0.5, -1.0, 15.0},
             {15.0, 1.0, 0.5, 40.0, -1.0},
         })
    {
        bool rejected = false;
        try
        {
            static_cast<void>(initial_stage_budget_ms(
                static_cast<int>(invalid[0]),
                static_cast<int>(invalid[1]),
                invalid[2],
                invalid[3],
                invalid[4]));
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        assert(rejected);
    }
}

void test_fallback_preflight_requires_setup_and_deadline_guards()
{
    using MDP::MSIRRT::fallback_solve_budget_ms;
    assert(fallback_solve_budget_ms(12.0, 15.0, 5.0) == 0.0);
    assert(fallback_solve_budget_ms(20.0, 15.0, 5.0) == 0.0);
    assert(std::abs(fallback_solve_budget_ms(40.0, 15.0, 5.0) - 20.0) < 1e-12);

    for (const auto invalid : std::vector<std::array<double, 3>>{
             {-1.0, 15.0, 5.0},
             {20.0, -1.0, 5.0},
             {20.0, 15.0, -1.0},
         })
    {
        bool rejected = false;
        try
        {
            static_cast<void>(fallback_solve_budget_ms(
                invalid[0], invalid[1], invalid[2]));
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        assert(rejected);
    }
}

void test_minimum_kinematic_arrival_frame_uses_joint_space_speed_bound()
{
    const int frame = MDP::MSIRRT::minimum_kinematic_arrival_frame(
        {0.0, 0.0}, {3.0, 4.0}, 10, 10, 2.0);
    assert(frame == 35);
}

void test_initial_horizon_selection_skips_infeasible_stage()
{
    const auto adjusted = MDP::MSIRRT::select_initial_planning_horizon(
        15, 30, 47, 92, 6, 15, true);
    assert(adjusted.raw_frame == 45);
    assert(adjusted.selected_frame == 60);
    assert(adjusted.adjusted_for_feasibility);
    assert(adjusted.skipped_infeasible_stages == 1);

    const auto legacy = MDP::MSIRRT::select_initial_planning_horizon(
        15, 30, 47, 92, 6, 15, false);
    assert(legacy.raw_frame == 45);
    assert(legacy.selected_frame == 45);
    assert(!legacy.adjusted_for_feasibility);
    assert(legacy.skipped_infeasible_stages == 0);

    const auto clamped = MDP::MSIRRT::select_initial_planning_horizon(
        15, 30, 53, 53, 6, 8, true);
    assert(clamped.selected_frame == 53);
    assert(clamped.adjusted_for_feasibility);
    assert(clamped.skipped_infeasible_stages == 1);
}

void test_conditional_exploration_feedback_tracks_rejection_pressure()
{
    MDP::MSIRRT::ConditionalExplorationFeedback feedback(
        0.10, 0.35, 32, 0.95, true);
    for (int index = 0; index < 31; ++index)
    {
        feedback.observe(true);
    }
    assert(std::abs(feedback.effective_rate() - 0.10) < 1e-12);
    feedback.observe(true);
    assert(std::abs(feedback.effective_rate() - 0.45) < 1e-12);
    assert(std::abs(feedback.rejection_ratio() - 1.0) < 1e-12);
    assert(feedback.trigger_count() == 1);

    feedback.reset_stage();
    assert(std::abs(feedback.effective_rate() - 0.10) < 1e-12);
    assert(std::abs(feedback.rejection_ratio()) < 1e-12);
    assert(feedback.trigger_count() == 1);
    for (int index = 0; index < 32; ++index)
    {
        feedback.observe(true);
    }
    assert(feedback.trigger_count() == 2);

    for (int index = 0; index < 16; ++index)
    {
        feedback.observe(false);
    }
    assert(std::abs(feedback.effective_rate() - 0.10) < 1e-12);
    assert(feedback.trigger_count() == 2);

    MDP::MSIRRT::ConditionalExplorationFeedback disabled(
        0.10, 0.90, 2, 0.50, false);
    disabled.observe(true);
    disabled.observe(true);
    assert(std::abs(disabled.effective_rate() - 0.10) < 1e-12);
    assert(disabled.trigger_count() == 0);
}

void test_online_stage_timing_reconciles_without_double_counting()
{
    MDP::MSIRRT::OnlineStageTiming timing;
    timing.prediction_ms = 1.0;
    timing.tube_update_ms = 2.0;
    timing.safe_interval_update_ms = 3.0;
    timing.tree_repair_ms = 4.0;
    timing.search_ms = 5.0;
    timing.fallback_ms = 6.0;

    assert(std::abs(timing.attributed_ms() - 21.0) < 1e-12);
    assert(timing.unattributed_ms(21.05) > 0.049);
    assert(timing.reconciles(20.91));
    assert(!timing.reconciles(20.89));

    timing.search_ms = -1.0;
    assert(!timing.reconciles(21.0));
    timing.search_ms = std::numeric_limits<double>::infinity();
    assert(!timing.reconciles(21.0));
}

void test_primary_failure_attribution_has_one_fixed_priority()
{
    using Category = MDP::MSIRRT::PrimaryFailureCategory;
    using MDP::MSIRRT::classify_primary_failure;
    using MDP::MSIRRT::primary_failure_category_name;

    MDP::MSIRRT::FailureAttributionInput failure;
    failure.task_completed = true;
    assert(classify_primary_failure(failure) == Category::None);

    failure = {};
    failure.result_valid = false;
    failure.safety_validation_failed = true;
    failure.deadline_missed = true;
    assert(classify_primary_failure(failure) == Category::InvalidOrMissingResult);

    failure = {};
    failure.safety_validation_failed = true;
    failure.deadline_missed = true;
    failure.prediction_unavailable = true;
    assert(classify_primary_failure(failure) == Category::SafetyValidationFailed);

    failure = {};
    failure.deadline_missed = true;
    failure.prediction_unavailable = true;
    failure.reliable_horizon_too_short = true;
    assert(classify_primary_failure(failure) == Category::DeadlineMissed);

    failure = {};
    failure.prediction_unavailable = true;
    failure.reliable_horizon_too_short = true;
    assert(classify_primary_failure(failure) == Category::PredictionUnavailableStop);

    failure = {};
    failure.reliable_horizon_too_short = true;
    assert(classify_primary_failure(failure) == Category::ReliableHorizonTooShort);

    failure = {};
    failure.terminal_update_index = 0;
    failure.ever_accepted_plan = false;
    assert(classify_primary_failure(failure) == Category::InitialPlanningFailed);

    failure = {};
    failure.terminal_update_index = 3;
    failure.ever_accepted_plan = true;
    assert(classify_primary_failure(failure) == Category::PostChangeRepairFailed);

    assert(std::string(primary_failure_category_name(Category::None)) == "none");
    assert(std::string(primary_failure_category_name(Category::InitialPlanningFailed)) ==
           "initial_planning_failed");
    assert(std::string(primary_failure_category_name(Category::PostChangeRepairFailed)) ==
           "post_change_repair_failed");
    assert(std::string(primary_failure_category_name(Category::ReliableHorizonTooShort)) ==
           "reliable_horizon_too_short");
    assert(std::string(primary_failure_category_name(Category::DeadlineMissed)) ==
           "deadline_missed");
    assert(std::string(primary_failure_category_name(Category::PredictionUnavailableStop)) ==
           "prediction_unavailable_stop");
    assert(std::string(primary_failure_category_name(Category::SafetyValidationFailed)) ==
           "safety_validation_failed");
    assert(std::string(primary_failure_category_name(Category::InvalidOrMissingResult)) ==
           "invalid_or_missing_result");
}

void test_update_zero_final_search_budget_is_isolated()
{
    using MDP::MSIRRT::select_search_continuation_budget;

    const auto baseline = select_search_continuation_budget(
        16.0, 15.0, 15.0, true, true, false);
    assert(baseline.effective_guard_ms == 15.0);
    assert(baseline.solve_budget_ms == 1.0);
    assert(baseline.allowed);

    const auto candidate = select_search_continuation_budget(
        16.0, 15.0, 5.0, true, true, false);
    assert(candidate.effective_guard_ms == 5.0);
    assert(candidate.solve_budget_ms == 11.0);
    assert(candidate.allowed);

    const auto post_change = select_search_continuation_budget(
        16.0, 15.0, 5.0, false, true, true);
    assert(post_change.effective_guard_ms == 15.0);
    assert(post_change.solve_budget_ms == 1.0);

    const auto not_maximum_horizon = select_search_continuation_budget(
        16.0, 15.0, 5.0, true, false, false);
    assert(not_maximum_horizon.effective_guard_ms == 15.0);

    const auto fallback_available = select_search_continuation_budget(
        16.0, 15.0, 5.0, true, true, true);
    assert(fallback_available.effective_guard_ms == 15.0);

    const auto exhausted = select_search_continuation_budget(
        5.0, 15.0, 5.0, true, true, false);
    assert(exhausted.solve_budget_ms == 0.0);
    assert(!exhausted.allowed);

    for (const auto invalid : std::vector<std::array<double, 3>>{
             {-1.0, 15.0, 5.0},
             {16.0, -1.0, 5.0},
             {16.0, 15.0, 4.9},
             {16.0, 15.0, 15.1},
             {std::numeric_limits<double>::quiet_NaN(), 15.0, 5.0},
         })
    {
        bool rejected = false;
        try
        {
            static_cast<void>(select_search_continuation_budget(
                invalid[0], invalid[1], invalid[2], true, true, false));
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        assert(rejected);
    }
}

} // namespace

int main()
{
    test_normalize_and_select_intervals();
    test_prediction_change_windows();
    test_interval_transition_classification();
    test_interval_set_transition_classification();
    test_temporal_dependency_index();
    test_reuse_rate();
    test_sequential_repair_keeps_only_original_survivors();
    test_motion_validation_includes_exact_integer_frame_configuration();
    test_execution_window_is_bounded_by_prediction_reliability_and_path();
    test_planning_budget_reserves_full_replan_time();
    test_initial_stage_budget_reuses_skipped_budget_with_guard();
    test_fallback_preflight_requires_setup_and_deadline_guards();
    test_minimum_kinematic_arrival_frame_uses_joint_space_speed_bound();
    test_initial_horizon_selection_skips_infeasible_stage();
    test_conditional_exploration_feedback_tracks_rejection_pressure();
    test_online_stage_timing_reconciles_without_double_counting();
    test_primary_failure_attribution_has_one_fixed_priority();
    test_update_zero_final_search_budget_is_isolated();
    std::cout << "repair type tests passed\n";
    return 0;
}
