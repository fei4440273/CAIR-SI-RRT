#include <coal/shape/geometric_shapes.h>
#include <vector>
#include <cassert>
#include <config_read_writer/config_read.hpp>
#include <config_read_writer/ObstacleJsonInfo.hpp>
#include <CollisionManager/ObjectObstacleFCL.hpp>

MDP::ObjectObstacleFCL::ObjectObstacleFCL(const std::vector<MDP::ObstacleCoordinate>& _positions, const std::string& _name, const std::string& _type, coal::ShapeBase* _collision_object, const bool& _is_static, std::vector<float> _security_margins) : positions(_positions), collision_object(_collision_object), name(_name), type(_type), is_static(_is_static), security_margins(std::move(_security_margins))
{
    transform.setQuatRotation(this->positions[0].rotation);
    transform.setTranslation(this->positions[0].pos);
}

MDP::ObjectObstacleFCL::~ObjectObstacleFCL()
{
}

void MDP::ObjectObstacleFCL::set_position(const unsigned int frame) const
{
    assert(frame >= 0);
    // std::cout<<frame<<' '<<positions.size()<<std::endl;
    assert(frame < positions.size());
    transform.setQuatRotation(positions[frame].rotation);
    transform.setTranslation(positions[frame].pos);
}
void MDP::ObjectObstacleFCL::set_position(const MDP::ObstacleCoordinate position) const
{
    transform.setQuatRotation(position.rotation);
    transform.setTranslation(position.pos);
}

coal::ShapeBase* MDP::ObjectObstacleFCL::get_collision_object()  const
{
    return this->collision_object;
}

coal::Transform3s MDP::ObjectObstacleFCL::get_transform() const
{
    return this->transform;
}

std::string MDP::ObjectObstacleFCL::get_name() const
{
    return this->name;
}

std::string MDP::ObjectObstacleFCL::get_type() const
{
    return this->type;
}

MDP::ObstacleCoordinate MDP::ObjectObstacleFCL::get_position(unsigned int frame) const
{
    assert(frame >= 0);
    assert(frame < this->positions.size());
    return this->positions[frame];
}

bool MDP::ObjectObstacleFCL::get_is_static() const
{
    return this->is_static;
}

float MDP::ObjectObstacleFCL::get_security_margin(unsigned int frame) const
{
    if (this->security_margins.empty())
    {
        return 0.0F;
    }
    if (this->security_margins.size() == 1)
    {
        return this->security_margins.front();
    }
    return this->security_margins.at(frame);
}

coal::CollisionGeometry *MDP::ObjectObstacleFCL::clone_collision_geometry(unsigned int) const
{
    return this->collision_object->clone();
}
