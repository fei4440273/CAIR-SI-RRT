#include <coal/shape/geometric_shapes.h>
#include <vector>
#include <cassert>
#include <config_read_writer/config_read.hpp>
#include <config_read_writer/SphereObstacleJsonInfo.hpp>
#include <CollisionManager/SphereObstacleFCL.hpp>
#include <CollisionManager/ObjectObstacleFCL.hpp>

MDP::SphereObstaclesFCL::SphereObstaclesFCL(const float _radius, const std::vector<MDP::ObstacleCoordinate> &_positions, std::string _name, const std::string& _type, const bool _is_static)
    : ObjectObstacleFCL(_positions, _name, _type, new coal::Sphere(_radius), _is_static),
      radius(_radius),
      sphere(static_cast<coal::Sphere *>(ObjectObstacleFCL::get_collision_object()))
{
}

MDP::SphereObstaclesFCL::SphereObstaclesFCL(const MDP::SphereObstacleJsonInfo &json_obstacle_info)

    : ObjectObstacleFCL(
          json_obstacle_info.get_coordinates(),
          json_obstacle_info.get_name(),
          json_obstacle_info.get_type(),
          new coal::Sphere(json_obstacle_info.get_radius()),
          json_obstacle_info.get_is_static(),
          json_obstacle_info.get_uncertainty_radii()),
      radius(json_obstacle_info.get_radius()),
      sphere(static_cast<coal::Sphere *>(ObjectObstacleFCL::get_collision_object()))
{
}

MDP::SphereObstaclesFCL::~SphereObstaclesFCL()
{
    delete this->sphere;
}

float MDP::SphereObstaclesFCL::get_radius() const
{
    return this->radius;
}

coal::CollisionGeometry *MDP::SphereObstaclesFCL::clone_collision_geometry(unsigned int frame) const
{
    return new coal::Sphere(this->radius + this->get_security_margin(frame));
}
