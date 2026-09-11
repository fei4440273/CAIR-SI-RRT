#include "TubeAblation.hpp"
#include "config_read_writer/SphereObstacleJsonInfo.hpp"
#include "config_read_writer/config_read.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{

const MDP::SphereObstacleJsonInfo &first_dynamic_sphere(
    const MDP::ConfigReader::SceneTask &scene)
{
    for (const auto &obstacle : scene.obstacles)
    {
        if (obstacle->get_type() == "dynamic_sphere")
        {
            return *static_cast<const MDP::SphereObstacleJsonInfo *>(obstacle.get());
        }
    }
    throw std::runtime_error("fixture has no dynamic sphere");
}

void assert_same_centers(
    const MDP::SphereObstacleJsonInfo &left,
    const MDP::SphereObstacleJsonInfo &right)
{
    const auto left_coordinates = left.get_coordinates();
    const auto right_coordinates = right.get_coordinates();
    assert(left_coordinates.size() == right_coordinates.size());
    for (std::size_t index = 0; index < left_coordinates.size(); ++index)
    {
        assert(left_coordinates[index].x == right_coordinates[index].x);
        assert(left_coordinates[index].y == right_coordinates[index].y);
        assert(left_coordinates[index].z == right_coordinates[index].z);
        assert(left_coordinates[index].quat_x == right_coordinates[index].quat_x);
        assert(left_coordinates[index].quat_y == right_coordinates[index].quat_y);
        assert(left_coordinates[index].quat_z == right_coordinates[index].quat_z);
        assert(left_coordinates[index].quat_w == right_coordinates[index].quat_w);
    }
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: test_tube_ablation <scene.json>\n";
        return 2;
    }

    MDP::ConfigReader reader(argv[1]);
    const auto source = reader.get_scene_task();
    const auto &source_sphere = first_dynamic_sphere(source);

    auto fixed = source;
    fixed.prediction_issue_frame = 15;
    fixed.reliable_until_frame = fixed.frame_count - 1;
    const MDP::MSIRRT::FixedTubeAblationConfig below_cap{0.07, 0.25, 30};
    const auto report = MDP::MSIRRT::apply_fixed_calibrated_tube(fixed, below_cap);
    assert(report.transformed_dynamic_spheres == 20);
    assert(report.reliable_until_frame == 45);
    assert(fixed.reliable_until_frame == 45);
    const auto &fixed_sphere = first_dynamic_sphere(fixed);
    assert_same_centers(source_sphere, fixed_sphere);
    assert(fixed_sphere.get_uncertainty_radii().size() ==
           source_sphere.get_coordinates().size());
    assert(std::abs(fixed_sphere.get_uncertainty_radius(15) -
                    source_sphere.get_uncertainty_radius(15)) < 1e-6F);
    assert(std::abs(fixed_sphere.get_uncertainty_radius(16) - 0.07F) < 1e-6F);

    auto above_cap = source;
    above_cap.prediction_issue_frame = 15;
    above_cap.reliable_until_frame = above_cap.frame_count - 1;
    const auto capped_report = MDP::MSIRRT::apply_fixed_calibrated_tube(
        above_cap,
        MDP::MSIRRT::FixedTubeAblationConfig{0.30, 0.25, 150});
    assert(capped_report.transformed_dynamic_spheres == 20);
    assert(capped_report.reliable_until_frame == 15);
    assert(above_cap.reliable_until_frame == 15);

    auto unavailable = source;
    unavailable.prediction_status = "unavailable";
    unavailable.prediction_issue_frame = 30;
    unavailable.reliable_until_frame = unavailable.frame_count - 1;
    MDP::MSIRRT::apply_fixed_calibrated_tube(unavailable, below_cap);
    assert(unavailable.reliable_until_frame == 30);

    bool rejected = false;
    try
    {
        auto invalid = source;
        MDP::MSIRRT::apply_fixed_calibrated_tube(
            invalid,
            MDP::MSIRRT::FixedTubeAblationConfig{-0.1, 0.25, 30});
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    assert(rejected);
    return 0;
}
