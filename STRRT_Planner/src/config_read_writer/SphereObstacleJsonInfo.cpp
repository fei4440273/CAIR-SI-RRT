#include <vector>
#include <string>
#include <config_read_writer/SphereObstacleJsonInfo.hpp>
#include <config_read_writer/ObstacleJsonInfo.hpp>
#include <cassert>
#include <cmath>
#include <stdexcept>
MDP::SphereObstacleJsonInfo::SphereObstacleJsonInfo(std::string _name, std::string _type,
                                                    const std::vector<std::vector<float>> &coordinates_raw,
                                                    float fps, float _radius,
                                                    bool _is_static,
                                                    std::vector<float> _uncertainty_radii)
    : MDP::ObstacleJsonInfo(_name, _type, coordinates_raw, fps, _is_static, MDP::ObstacleJsonInfo::ObstacleType::SPHERE),
      radius(_radius), uncertainty_radii(std::move(_uncertainty_radii))
{
    const std::size_t expected_frames = _is_static ? 1 : coordinates_raw.size();
    if (!uncertainty_radii.empty() && uncertainty_radii.size() != 1 && uncertainty_radii.size() != expected_frames)
    {
        throw std::invalid_argument("uncertainty_radii must contain one value or one value per obstacle frame");
    }
    for (const float value : uncertainty_radii)
    {
        if (!std::isfinite(value) || value < 0.0F)
        {
            throw std::invalid_argument("uncertainty radius must be finite and nonnegative");
        }
    }
}

float MDP::SphereObstacleJsonInfo::get_radius() const
{
    return this->radius;
}

const std::vector<float> &MDP::SphereObstacleJsonInfo::get_uncertainty_radii() const
{
    return this->uncertainty_radii;
}

float MDP::SphereObstacleJsonInfo::get_uncertainty_radius(std::size_t frame) const
{
    if (this->uncertainty_radii.empty())
    {
        return 0.0F;
    }
    if (this->uncertainty_radii.size() == 1)
    {
        return this->uncertainty_radii.front();
    }
    return this->uncertainty_radii.at(frame);
}
