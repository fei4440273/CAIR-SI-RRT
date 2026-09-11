#include "TubeAblation.hpp"

#include "config_read_writer/SphereObstacleJsonInfo.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{

std::vector<std::vector<float>> sphere_coordinates(
    const MDP::SphereObstacleJsonInfo &sphere)
{
    std::vector<std::vector<float>> coordinates;
    for (const auto &position : sphere.get_coordinates())
    {
        coordinates.push_back({
            position.x,
            position.y,
            position.z,
            position.quat_x,
            position.quat_y,
            position.quat_z,
            position.quat_w});
    }
    return coordinates;
}

} // namespace

MDP::MSIRRT::FixedTubeAblationReport MDP::MSIRRT::apply_fixed_calibrated_tube(
    MDP::ConfigReader::SceneTask &scene,
    const FixedTubeAblationConfig &config)
{
    if (!std::isfinite(config.fixed_radius) || config.fixed_radius < 0.0 ||
        !std::isfinite(config.maximum_allowed_radius) ||
        config.maximum_allowed_radius < 0.0 ||
        config.maximum_prediction_horizon_frames < 0)
    {
        throw std::invalid_argument("fixed-tube ablation configuration is invalid");
    }
    if (scene.frame_count == 0 || scene.prediction_issue_frame >= scene.frame_count)
    {
        throw std::invalid_argument("fixed-tube ablation scene metadata is invalid");
    }

    FixedTubeAblationReport report;
    for (std::size_t obstacle_id = 0; obstacle_id < scene.obstacles.size(); ++obstacle_id)
    {
        const auto &obstacle = scene.obstacles[obstacle_id];
        if (obstacle->get_type() != "dynamic_sphere")
        {
            continue;
        }
        if (obstacle->get_obstacle_type() !=
            MDP::ObstacleJsonInfo::ObstacleType::SPHERE)
        {
            throw std::runtime_error("dynamic_sphere has a non-sphere obstacle type");
        }
        const auto *sphere = static_cast<const MDP::SphereObstacleJsonInfo *>(
            obstacle.get());
        const auto coordinates = sphere_coordinates(*sphere);
        std::vector<float> fixed_future_radii(coordinates.size(), 0.0F);
        for (std::size_t frame = 0; frame < fixed_future_radii.size(); ++frame)
        {
            fixed_future_radii[frame] = frame <= scene.prediction_issue_frame
                ? sphere->get_uncertainty_radius(frame)
                : static_cast<float>(config.fixed_radius);
        }
        scene.obstacles[obstacle_id] =
            std::make_shared<MDP::SphereObstacleJsonInfo>(
                sphere->get_name(),
                sphere->get_type(),
                coordinates,
                static_cast<float>(scene.fps),
                sphere->get_radius(),
                false,
                fixed_future_radii);
        ++report.transformed_dynamic_spheres;
    }
    if (report.transformed_dynamic_spheres == 0)
    {
        throw std::runtime_error("fixed-tube ablation requires a dynamic sphere");
    }

    const int issue_frame = static_cast<int>(scene.prediction_issue_frame);
    int reliable_until = std::min(
        static_cast<int>(scene.frame_count) - 1,
        issue_frame + config.maximum_prediction_horizon_frames);
    if (scene.prediction_status == "unavailable" ||
        config.fixed_radius > config.maximum_allowed_radius)
    {
        reliable_until = issue_frame;
    }
    scene.reliable_until_frame = static_cast<unsigned int>(reliable_until);
    report.reliable_until_frame = reliable_until;
    return report;
}
