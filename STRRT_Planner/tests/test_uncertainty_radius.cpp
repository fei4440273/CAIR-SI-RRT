#include "config_read_writer/SphereObstacleJsonInfo.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::vector<float>> coordinates(int frame_count)
{
    return std::vector<std::vector<float>>(
        frame_count,
        std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F});
}

void test_missing_radii_are_zero()
{
    MDP::SphereObstacleJsonInfo sphere(
        "sphere", "dynamic_sphere", coordinates(3), 30.0F, 0.1F, false);
    assert(sphere.get_uncertainty_radii().empty());
    assert(sphere.get_uncertainty_radius(0) == 0.0F);
    assert(sphere.get_uncertainty_radius(2) == 0.0F);
}

void test_single_radius_is_broadcast()
{
    MDP::SphereObstacleJsonInfo sphere(
        "sphere", "dynamic_sphere", coordinates(3), 30.0F, 0.1F, false, {0.04F});
    assert(std::abs(sphere.get_uncertainty_radius(0) - 0.04F) < 1e-7F);
    assert(std::abs(sphere.get_uncertainty_radius(2) - 0.04F) < 1e-7F);
}

void test_per_frame_radii()
{
    MDP::SphereObstacleJsonInfo sphere(
        "sphere", "dynamic_sphere", coordinates(3), 30.0F, 0.1F, false,
        {0.01F, 0.02F, 0.03F});
    assert(std::abs(sphere.get_uncertainty_radius(1) - 0.02F) < 1e-7F);
}

void test_invalid_radii_are_rejected()
{
    bool wrong_length_rejected = false;
    try
    {
        MDP::SphereObstacleJsonInfo sphere(
            "sphere", "dynamic_sphere", coordinates(3), 30.0F, 0.1F, false,
            {0.01F, 0.02F});
    }
    catch (const std::invalid_argument &)
    {
        wrong_length_rejected = true;
    }
    assert(wrong_length_rejected);

    bool negative_rejected = false;
    try
    {
        MDP::SphereObstacleJsonInfo sphere(
            "sphere", "dynamic_sphere", coordinates(3), 30.0F, 0.1F, false,
            {-0.01F});
    }
    catch (const std::invalid_argument &)
    {
        negative_rejected = true;
    }
    assert(negative_rejected);
}

} // namespace

int main()
{
    test_missing_radii_are_zero();
    test_single_radius_is_broadcast();
    test_per_frame_radii();
    test_invalid_radii_are_rejected();
    std::cout << "uncertainty radius tests passed\n";
    return 0;
}
