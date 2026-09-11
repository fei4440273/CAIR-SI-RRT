#pragma once

#include "config_read_writer/config_read.hpp"

#include <cstddef>

namespace MDP::MSIRRT
{

struct FixedTubeAblationConfig
{
    double fixed_radius = 0.0;
    double maximum_allowed_radius = 0.0;
    int maximum_prediction_horizon_frames = 0;
};

struct FixedTubeAblationReport
{
    std::size_t transformed_dynamic_spheres = 0;
    int reliable_until_frame = 0;
};

FixedTubeAblationReport apply_fixed_calibrated_tube(
    MDP::ConfigReader::SceneTask &scene,
    const FixedTubeAblationConfig &config);

} // namespace MDP::MSIRRT
