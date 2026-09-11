#include <algorithm>
#include <vector>
#include <CollisionManager/CollisionManager.hpp>
#include <CollisionManager/ObjectObstacleFCL.hpp>
#include <CollisionManager/CubeObstacleFCL.hpp>
#include <CollisionManager/SphereObstacleFCL.hpp>
#include <CollisionManager/RobotObstacleFCL.hpp>
#include <iostream>
#include <coal/math/transform.h>
#include <coal/collision_data.h>
#include <coal/collision_object.h>
#include <coal/broadphase/default_broadphase_callbacks.h>
#include <coal/shape/convex.h>
#include "config_read_writer/ResultsWriter.hpp"
#include "coal/broadphase/broadphase_callbacks.h"
#include <random>
#include <fstream>
#include <cstdlib>
#include <set>
#include <unordered_set>

namespace
{

bool same_coordinate(const MDP::ObstacleCoordinate &left, const MDP::ObstacleCoordinate &right)
{
    return left.x == right.x && left.y == right.y && left.z == right.z &&
           left.quat_x == right.quat_x && left.quat_y == right.quat_y &&
           left.quat_z == right.quat_z && left.quat_w == right.quat_w;
}

void delete_obstacle(MDP::ObjectObstacleFCL *obstacle)
{
    const std::string type = obstacle->get_type();
    if (type == "static_box" || type == "dynamic_box")
    {
        delete static_cast<MDP::CubeObstaclesFCL *>(obstacle);
    }
    else if (type == "static_sphere" || type == "dynamic_sphere")
    {
        delete static_cast<MDP::SphereObstaclesFCL *>(obstacle);
    }
    else
    {
        throw std::invalid_argument("unsupported obstacle type in collision update");
    }
}

MDP::ObjectObstacleFCL *make_obstacle(
    const std::shared_ptr<MDP::ObstacleJsonInfo> &obstacle)
{
    if (obstacle->get_obstacle_type() == MDP::ObstacleJsonInfo::ObstacleType::BOX)
    {
        return new MDP::CubeObstaclesFCL(
            *static_cast<const MDP::CubeObstacleJsonInfo *>(obstacle.get()));
    }
    if (obstacle->get_obstacle_type() == MDP::ObstacleJsonInfo::ObstacleType::SPHERE)
    {
        return new MDP::SphereObstaclesFCL(
            *static_cast<const MDP::SphereObstacleJsonInfo *>(obstacle.get()));
    }
    throw std::invalid_argument("unsupported obstacle geometry in collision update");
}

bool same_obstacle_geometry(
    const std::shared_ptr<MDP::ObstacleJsonInfo> &before,
    const std::shared_ptr<MDP::ObstacleJsonInfo> &after)
{
    if (before->get_name() != after->get_name() ||
        before->get_type() != after->get_type() ||
        before->get_obstacle_type() != after->get_obstacle_type() ||
        before->get_is_static() != after->get_is_static())
    {
        return false;
    }
    if (before->get_obstacle_type() == MDP::ObstacleJsonInfo::ObstacleType::SPHERE)
    {
        return static_cast<const MDP::SphereObstacleJsonInfo *>(before.get())->get_radius() ==
               static_cast<const MDP::SphereObstacleJsonInfo *>(after.get())->get_radius();
    }
    return static_cast<const MDP::CubeObstacleJsonInfo *>(before.get())->get_dimensions() ==
           static_cast<const MDP::CubeObstacleJsonInfo *>(after.get())->get_dimensions();
}

bool topology_compatible(
    const MDP::ConfigReader::SceneTask &before,
    const MDP::ConfigReader::SceneTask &after)
{
    if (before.frame_count != after.frame_count || before.fps != after.fps ||
        before.obstacles.size() != after.obstacles.size() ||
        !before.robot_obstacles.empty() || !after.robot_obstacles.empty() ||
        before.robot_urdf_path != after.robot_urdf_path ||
        before.robot_base_coords != after.robot_base_coords ||
        before.robot_base_quat_rot != after.robot_base_quat_rot ||
        before.robot_joints_order != after.robot_joints_order ||
        before.robot_capsules_radius != after.robot_capsules_radius ||
        before.robot_joint_count != after.robot_joint_count ||
        before.robot_base_position_vector.size() != after.robot_base_position_vector.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < before.robot_base_position_vector.size(); ++index)
    {
        if (!same_coordinate(
                before.robot_base_position_vector[index],
                after.robot_base_position_vector[index]))
        {
            return false;
        }
    }
    for (std::size_t index = 0; index < before.obstacles.size(); ++index)
    {
        if (!same_obstacle_geometry(before.obstacles[index], after.obstacles[index]))
        {
            return false;
        }
        const std::size_t required = before.obstacles[index]->get_is_static()
            ? 1
            : before.frame_count;
        if (before.obstacles[index]->get_coordinates().size() < required ||
            after.obstacles[index]->get_coordinates().size() < required)
        {
            return false;
        }
    }
    return true;
}

} // namespace

MDP::CollisionManager::CollisionManager(const MDP::ConfigReader::SceneTask &_scene_task) : planned_robot(_scene_task.robot_urdf_path, _scene_task.robot_base_position_vector, _scene_task.robot_joints_order, std::vector<MDP::RobotObstacleJsonInfo::PathState>(),_scene_task,false),
                                                                                          scene_task(_scene_task)
{ // init robot (in initializer list)

    MDP::ResultsWriter::get_instance().restart_collision_checker();                                         

    // init obstacles
    for (auto obstacle : scene_task.obstacles)
    { // TODO: divide obstacles into static and dynamic one.
        if (obstacle->get_type() == "static_box")
        {
            this->obstacles.push_back(static_cast<MDP::ObjectObstacleFCL *>(new MDP::CubeObstaclesFCL(*static_cast<CubeObstacleJsonInfo *>(obstacle.get()))));
        }
        else if (obstacle->get_type() == "dynamic_box")
        {
            this->obstacles.push_back(static_cast<MDP::ObjectObstacleFCL *>(new MDP::CubeObstaclesFCL(*static_cast<CubeObstacleJsonInfo *>(obstacle.get()))));
        }
        else if (obstacle->get_type() == "static_sphere")
        {
            this->obstacles.push_back(static_cast<MDP::ObjectObstacleFCL *>(new MDP::SphereObstaclesFCL(*static_cast<SphereObstacleJsonInfo *>(obstacle.get()))));
        }
        else if (obstacle->get_type() == "dynamic_sphere")
        {
            this->obstacles.push_back(static_cast<MDP::ObjectObstacleFCL *>(new MDP::SphereObstaclesFCL(*static_cast<SphereObstacleJsonInfo *>(obstacle.get()))));
        }
    }

    this->robot_obstacle_all_joint_count = 0;
    for (MDP::RobotObstacleJsonInfo obstacle : scene_task.robot_obstacles)
    {
        if (obstacle.get_type() == "dynamic_robot" | obstacle.get_type() == "static_robot")
        {
            this->robot_obstacles.emplace_back(obstacle.get_urdf_file_path(), obstacle.get_base_coordinates(), obstacle.get_robot_joints_order(), obstacle.get_trajectory(),this->scene_task,obstacle.get_is_static());
            this->robot_obstacle_all_joint_count += this->robot_obstacles.back().get_geometric_shapes().size();
        }
    }

    this->frame_count = this->scene_task.frame_count;
    this->collision_object_count = this->obstacles.size();
    this->safe_interval_indexed_until.assign(
        this->collision_object_count, this->frame_count - 1);
    this->minimum_indexed_until_frame = this->frame_count - 1;

    this->obstacle_security_margins.resize(this->collision_object_count);
    for (int obstacle_id = 0; obstacle_id < this->collision_object_count; ++obstacle_id)
    {
        this->obstacle_security_margins[obstacle_id].reserve(this->frame_count);
        for (int frame = 0; frame < this->frame_count; ++frame)
        {
            this->obstacle_security_margins[obstacle_id].push_back(
                this->obstacles[obstacle_id]->get_security_margin(frame));
        }
    }

    // fill positions array
    this->positions = new coal::Transform3s[frame_count * collision_object_count];

    for (int obstacle_id = 0; obstacle_id < collision_object_count; obstacle_id++)
    {
        MDP::ObjectObstacleFCL *obstacle = this->obstacles[obstacle_id];
        for (int frame = 0; frame < this->frame_count; frame++)
        {
            if (obstacle->get_is_static())
            {
                MDP::ObstacleCoordinate obj_position = obstacle->get_position(0);
                // frame* collision_object_count is frame offset
                // obstacle_id is object offset
                this->positions[frame * collision_object_count + obstacle_id].setQuatRotation(obj_position.rotation);
                this->positions[frame * collision_object_count + obstacle_id].setTranslation(obj_position.pos);
            }
            else
            {
                MDP::ObstacleCoordinate obj_position = obstacle->get_position(frame);
                // frame* collision_object_count is frame offset
                // obstacle_id is object offset
                this->positions[frame * collision_object_count + obstacle_id].setQuatRotation(obj_position.rotation);
                this->positions[frame * collision_object_count + obstacle_id].setTranslation(obj_position.pos);
            }
        }
    }
    // fill collision pairs
    std::vector<MDP::RobotObstacleFCL::JointCollisionObject> robot_joints = this->planned_robot.get_collision_object_for_robot_angles(scene_task.start_configuration);
    this->collision_robot_links_count = robot_joints.size();
    this->robot_joint_offset = 0;

    for (int robot_joint_id = 0; robot_joint_id < this->collision_robot_links_count; robot_joint_id++)
    {
        for (int obstacle_id = 0; obstacle_id < collision_object_count; obstacle_id++)
        {
            MDP::ObjectObstacleFCL *obstacle = this->obstacles[obstacle_id];
            this->collision_pairs.emplace_back(robot_joints[robot_joint_id + this->robot_joint_offset].collision_object.get(), obstacle->get_collision_object());
        }
        for (int obstacle_robot_id = 0; obstacle_robot_id < this->robot_obstacles.size(); obstacle_robot_id++)
        {
            std::vector<std::shared_ptr<coal::ShapeBase>> joint_objects = this->robot_obstacles[obstacle_robot_id].get_geometric_shapes();
            for (std::shared_ptr<coal::ShapeBase> joint : joint_objects)
            {
                this->collision_pairs.emplace_back(robot_joints[robot_joint_id + this->robot_joint_offset].collision_object.get(), joint.get());
            }
        }
    }

    // construct safeinterval broadphase colliison manager
    MDP::ResultsWriter::get_instance().safe_intervals_init_time.reset();
    coal::DynamicAABBTreeArrayCollisionManager *temp = new coal::DynamicAABBTreeArrayCollisionManager();
    temp->tree_init_level = 2;
    this->broadphase_manager = temp;
    // get all spheres for all frames_and_ids and set their time
    this->frames_and_ids_safe_interval = new std::pair<std::pair<int, int>, int>[ scene_task.frame_count *(collision_object_count + this->robot_obstacle_all_joint_count)];
    this->frames_and_ids = new std::pair<int, int>[  scene_task.frame_count *(collision_object_count + this->robot_obstacle_all_joint_count)];


    // Register scene obstacles into the cross-frame broadphase used by
    // get_safe_intervals. For static obstacles we emit one entry covering the
    // whole frame range; for dynamic obstacles we compress consecutive frames
    // where position/rotation don't change (same scheme as robot obstacles).
    {
        const int total_obs_count = collision_object_count + this->robot_obstacle_all_joint_count;
        for (int obstacle_id = 0; obstacle_id < collision_object_count; obstacle_id++)
        {
            MDP::ObjectObstacleFCL *obstacle = this->obstacles[obstacle_id];
            auto emit_window = [&](int starting_static_frame, int ending_static_frame,
                                    const MDP::ObstacleCoordinate &obj_position) {
                this->all_spheres.push_back(new coal::CollisionObject(
                    std::shared_ptr<coal::CollisionGeometry>(obstacle->clone_collision_geometry(starting_static_frame)),
                    obj_position.rotation.toRotationMatrix(),
                    obj_position.pos));
                auto *slot = this->frames_and_ids_safe_interval
                             + starting_static_frame * total_obs_count + obstacle_id;
                slot->first = std::pair<int, int>(starting_static_frame, ending_static_frame);
                slot->second = obstacle_id;
                this->all_spheres.back()->setUserData(slot);
                this->all_spheres.back()->collisionGeometry()->computeLocalAABB();
            };

            if (obstacle->get_is_static())
            {
                emit_window(0, this->frame_count - 1, obstacle->get_position(0));
            }
            else
            {
                for (int frame = 0; frame < this->frame_count; frame++)
                {
                    int starting_static_frame = frame;
                    int ending_static_frame = frame;
                    MDP::ObstacleCoordinate base_pos = obstacle->get_position(frame);
                    for (int sf = frame + 1; sf < this->frame_count; sf++)
                    {
                        MDP::ObstacleCoordinate p = obstacle->get_position(sf);
                        if (p.x != base_pos.x || p.y != base_pos.y || p.z != base_pos.z
                            || p.quat_x != base_pos.quat_x || p.quat_y != base_pos.quat_y
                            || p.quat_z != base_pos.quat_z || p.quat_w != base_pos.quat_w
                            || obstacle->get_security_margin(sf) != obstacle->get_security_margin(starting_static_frame))
                        {
                            break;
                        }
                        ending_static_frame = sf;
                    }
                    emit_window(starting_static_frame, ending_static_frame, base_pos);
                    frame = ending_static_frame;
                }
            }
        }
    }

    int total_joint_id = 0;
    for (int obstacle_robot_id = 0; obstacle_robot_id < this->robot_obstacles.size(); obstacle_robot_id++)
    {
        int curr_joint_count = 0;
        for (int frame = 0; frame < this->frame_count; frame++)
        {       
            int starting_static_frame = frame;
            int ending_static_frame = frame;

            std::vector<MDP::RobotObstacleFCL::JointCollisionObject> static_result_obstacle_joints = this->robot_obstacles[obstacle_robot_id].get_collision_object_for_frame(frame);
            curr_joint_count = static_result_obstacle_joints.size();
            for (int static_frame = starting_static_frame; static_frame < this->frame_count; static_frame++)
            {
                bool is_static = true;
                std::vector<MDP::RobotObstacleFCL::JointCollisionObject> promising_static_obstacle_joints = this->robot_obstacles[obstacle_robot_id].get_collision_object_for_frame(static_frame);
                for (int joint_id = 0; joint_id < promising_static_obstacle_joints.size(); joint_id++)
                {
                    if (promising_static_obstacle_joints[joint_id].transform != static_result_obstacle_joints[joint_id].transform){
                        is_static = false;
                        break;
                    }
                }
                if (is_static){
                    ending_static_frame = static_frame;
                }
                else{
                    break;
                }
            }
            frame = ending_static_frame;

            // std::cout<<"starting_static_frame: "<<starting_static_frame<<" ending_static_frame: "<<ending_static_frame<< std::endl;
            for (int joint_id = 0; joint_id < static_result_obstacle_joints.size(); joint_id++)
            {
                this->all_spheres.push_back(new coal::CollisionObject(std::shared_ptr<coal::CollisionGeometry>(static_result_obstacle_joints[joint_id].collision_object.get()->clone()), static_result_obstacle_joints[joint_id].transform));
                this->frames_and_ids_safe_interval[starting_static_frame *(collision_object_count + this->robot_obstacle_all_joint_count) +(this->collision_object_count + total_joint_id + joint_id)].first = std::pair<int, int>(starting_static_frame, ending_static_frame);
                this->frames_and_ids_safe_interval[starting_static_frame *(collision_object_count + this->robot_obstacle_all_joint_count) +(this->collision_object_count + total_joint_id + joint_id)].second = this->collision_object_count + total_joint_id + joint_id;
                this->all_spheres.back()->setUserData(this->frames_and_ids_safe_interval + starting_static_frame *(collision_object_count + this->robot_obstacle_all_joint_count) +(this->collision_object_count + total_joint_id + joint_id));
                this->all_spheres.back()->collisionGeometry()->computeLocalAABB();
            }
        }
        total_joint_id += curr_joint_count;
    }

    this->broadphase_manager->registerObjects(this->all_spheres);
    this->broadphase_manager->setup();
    MDP::ResultsWriter::get_instance().safe_intervals_init_time.pause_timer();

    MDP::ResultsWriter::get_instance().collision_check_init_time.reset();

    for (int frame = 0; frame < this->scene_task.frame_count; frame++)
    {

        this->by_frame_spheres.emplace_back();
        for (int obstacle_id = 0; obstacle_id < collision_object_count; obstacle_id++)
        {
            MDP::ObjectObstacleFCL *obstacle = this->obstacles[obstacle_id];
            if (obstacle->get_is_static())
            {
                MDP::ObstacleCoordinate obj_position = obstacle->get_position(0);
                this->by_frame_spheres[frame].push_back(new coal::CollisionObject(std::shared_ptr<coal::CollisionGeometry>(obstacle->clone_collision_geometry(0)), obj_position.rotation.toRotationMatrix(), obj_position.pos));
                this->frames_and_ids[frame + scene_task.frame_count * obstacle_id].first = frame;
                this->frames_and_ids[frame + scene_task.frame_count * obstacle_id].second = obstacle_id;
                // this->all_spheres.back()->setUserData(this->frames_and_ids + frame + scene_task.frame_count * obstacle_id);
                this->by_frame_spheres[frame].back()->collisionGeometry()->computeLocalAABB();
                this->by_frame_spheres[frame].back()->setUserData(this->frames_and_ids + frame + scene_task.frame_count * obstacle_id);
            }
            else
            {

                MDP::ObstacleCoordinate obj_position = obstacle->get_position(frame);
                this->by_frame_spheres[frame].push_back(new coal::CollisionObject(std::shared_ptr<coal::CollisionGeometry>(obstacle->clone_collision_geometry(frame)), obj_position.rotation.toRotationMatrix(), obj_position.pos));

                this->frames_and_ids[frame + scene_task.frame_count * obstacle_id].first = frame;
                this->frames_and_ids[frame + scene_task.frame_count * obstacle_id].second = obstacle_id;
                // this->all_spheres.back()->setUserData(this->frames_and_ids + frame + scene_task.frame_count * obstacle_id);
                this->by_frame_spheres[frame].back()->collisionGeometry()->computeLocalAABB();
                this->by_frame_spheres[frame].back()->setUserData(this->frames_and_ids + frame + scene_task.frame_count * obstacle_id);
            }
        }
        int total_joint_id = 0;

        for (int obstacle_robot_id = 0; obstacle_robot_id < this->robot_obstacles.size(); obstacle_robot_id++)
        {

            std::vector<MDP::RobotObstacleFCL::JointCollisionObject> result_obstacle_joints = this->robot_obstacles[obstacle_robot_id].get_collision_object_for_frame(frame);
            for (int joint_id = 0; joint_id < result_obstacle_joints.size(); joint_id++)
            {
                this->by_frame_spheres[frame].push_back(new coal::CollisionObject(std::shared_ptr<coal::CollisionGeometry>(result_obstacle_joints[joint_id].collision_object.get()->clone()), result_obstacle_joints[joint_id].transform));
                this->by_frame_spheres[frame].back()->collisionGeometry()->computeLocalAABB();
                this->frames_and_ids[frame + scene_task.frame_count * (this->collision_object_count + total_joint_id + joint_id)].first = frame;
                this->frames_and_ids[frame + scene_task.frame_count * (this->collision_object_count + total_joint_id + joint_id)].second =  this->collision_object_count + total_joint_id + joint_id;
                this->by_frame_spheres[frame].back()->setUserData(this->frames_and_ids + frame + scene_task.frame_count * (this->collision_object_count + total_joint_id + joint_id));
            }

            total_joint_id += result_obstacle_joints.size();
        }
        

        this->frame_broadphase_managers.push_back(new coal::DynamicAABBTreeArrayCollisionManager());
        static_cast<coal::DynamicAABBTreeArrayCollisionManager *>((this->frame_broadphase_managers[frame]))->tree_init_level = 2;

        this->frame_broadphase_managers[frame]->registerObjects(this->by_frame_spheres[frame]);
        this->frame_broadphase_managers[frame]->setup();
    }
    MDP::ResultsWriter::get_instance().collision_check_init_time.pause_timer();

}

MDP::CollisionManager::~CollisionManager()
{
    for (int i = 0; i < this->obstacles.size(); i++)
    {
        // Check the type of obstacle and cast appropriately before deletion
        std::string obstacle_type = this->obstacles[i]->get_type();
        if (obstacle_type == "static_box" || obstacle_type == "dynamic_box")
        {
            delete static_cast<MDP::CubeObstaclesFCL*>(this->obstacles[i]);
        }
        else if (obstacle_type == "static_sphere" || obstacle_type == "dynamic_sphere")
        {
            delete static_cast<MDP::SphereObstaclesFCL*>(this->obstacles[i]);
        }
        else
        {
            assert(false);//undefined behaviour
            delete this->obstacles[i];
        }
    }

    this->obstacles.clear();

    for (int i = 0; i < this->all_spheres.size(); i++)
    {
        delete this->all_spheres[i];
    }

    for (int i = 0; i < this->by_frame_spheres.size(); i++)
    {
        for (int j = 0; j < this->by_frame_spheres[i].size(); j++)
        {
            delete this->by_frame_spheres[i][j];
        }
    }

    for (int i = 0; i < this->frame_broadphase_managers.size(); i++)
    {
        delete this->frame_broadphase_managers[i];
    }

    this->all_spheres.clear();

    delete[] this->positions;
    delete[] this->frames_and_ids;
    delete[] this->frames_and_ids_safe_interval;
    delete this->broadphase_manager;
}

MDP::CollisionManager::CollisionUpdateReport MDP::CollisionManager::update_scene(
    const MDP::ConfigReader::SceneTask &updated_scene,
    const std::vector<std::pair<int, int>> &changed_windows)
{
    return this->update_scene(
        updated_scene, changed_windows, this->frame_count - 1);
}

MDP::CollisionManager::CollisionUpdateReport MDP::CollisionManager::update_scene(
    const MDP::ConfigReader::SceneTask &updated_scene,
    const std::vector<std::pair<int, int>> &changed_windows,
    int maximum_indexed_frame)
{
    CollisionUpdateReport report;
    if (!topology_compatible(this->scene_task, updated_scene))
    {
        report.full_rebuild_fallback = true;
        return report;
    }
    if (maximum_indexed_frame < 0 || maximum_indexed_frame >= this->frame_count)
    {
        throw std::out_of_range("collision-update maximum frame is outside the scene");
    }
    const int previously_indexed_until_frame = this->minimum_indexed_until_frame;

    std::set<int> requested_frames;
    for (const auto &[unclamped_first, unclamped_last] : changed_windows)
    {
        const int first = std::max(0, unclamped_first);
        const int last = std::min(maximum_indexed_frame, unclamped_last);
        for (int frame = first; frame <= last; ++frame)
        {
            requested_frames.insert(frame);
        }
    }
    for (int frame = previously_indexed_until_frame + 1;
         frame <= maximum_indexed_frame;
         ++frame)
    {
        requested_frames.insert(frame);
    }
    this->minimum_indexed_until_frame = maximum_indexed_frame;

    std::unordered_set<int> changed_obstacle_ids;
    for (int obstacle_id = 0; obstacle_id < this->collision_object_count; ++obstacle_id)
    {
        const auto &before = this->scene_task.obstacles[obstacle_id];
        const auto &after = updated_scene.obstacles[obstacle_id];
        const auto &before_coordinates = before->get_coordinates();
        const auto &after_coordinates = after->get_coordinates();
        const auto *before_sphere = before->get_obstacle_type() == MDP::ObstacleJsonInfo::ObstacleType::SPHERE
            ? static_cast<const MDP::SphereObstacleJsonInfo *>(before.get())
            : nullptr;
        const auto *after_sphere = after->get_obstacle_type() == MDP::ObstacleJsonInfo::ObstacleType::SPHERE
            ? static_cast<const MDP::SphereObstacleJsonInfo *>(after.get())
            : nullptr;
        const auto frame_changed = [&](int frame) {
            const int before_frame = before->get_is_static() ? 0 : frame;
            const int after_frame = after->get_is_static() ? 0 : frame;
            return !same_coordinate(
                       before_coordinates[before_frame], after_coordinates[after_frame]) ||
                (before_sphere != nullptr &&
                 before_sphere->get_uncertainty_radius(frame) !=
                     after_sphere->get_uncertainty_radius(frame));
        };
        for (int frame = 0; frame < this->frame_count; ++frame)
        {
            if (!frame_changed(frame))
            {
                continue;
            }
            if (frame <= maximum_indexed_frame)
            {
                requested_frames.insert(frame);
                changed_obstacle_ids.insert(obstacle_id);
            }
            else
            {
                this->safe_interval_indexed_until[obstacle_id] = std::min(
                    this->safe_interval_indexed_until[obstacle_id],
                    maximum_indexed_frame);
            }
        }
        if (this->safe_interval_indexed_until[obstacle_id] < maximum_indexed_frame)
        {
            changed_obstacle_ids.insert(obstacle_id);
        }
    }

    if (changed_obstacle_ids.empty())
    {
        auto replacement_scene = updated_scene;
        this->scene_task = std::move(replacement_scene);
        return report;
    }
    report.frames_rebuilt = requested_frames.size();
    report.obstacles_reindexed = changed_obstacle_ids.size();

    delete this->broadphase_manager;
    this->broadphase_manager = nullptr;
    for (auto iterator = this->all_spheres.begin(); iterator != this->all_spheres.end();)
    {
        auto *metadata = static_cast<std::pair<std::pair<int, int>, int> *>((*iterator)->getUserData());
        if (changed_obstacle_ids.count(metadata->second) != 0)
        {
            delete *iterator;
            iterator = this->all_spheres.erase(iterator);
        }
        else
        {
            ++iterator;
        }
    }

    for (const int frame : requested_frames)
    {
        delete this->frame_broadphase_managers[frame];
        this->frame_broadphase_managers[frame] = nullptr;
        for (const int obstacle_id : changed_obstacle_ids)
        {
            auto *previous = this->by_frame_spheres[frame][obstacle_id];
            delete previous;
            this->by_frame_spheres[frame][obstacle_id] = nullptr;
        }
    }

    this->collision_pairs.clear();
    for (const int obstacle_id : changed_obstacle_ids)
    {
        delete_obstacle(this->obstacles[obstacle_id]);
        this->obstacles[obstacle_id] = make_obstacle(updated_scene.obstacles[obstacle_id]);
        this->obstacle_security_margins[obstacle_id].clear();
        this->obstacle_security_margins[obstacle_id].reserve(this->frame_count);
        for (int frame = 0; frame < this->frame_count; ++frame)
        {
            this->obstacle_security_margins[obstacle_id].push_back(
                this->obstacles[obstacle_id]->get_security_margin(frame));
            const auto position = this->obstacles[obstacle_id]->get_position(
                this->obstacles[obstacle_id]->get_is_static() ? 0 : frame);
            this->positions[frame * this->collision_object_count + obstacle_id].setQuatRotation(
                position.rotation);
            this->positions[frame * this->collision_object_count + obstacle_id].setTranslation(
                position.pos);
        }
    }

    for (const int frame : requested_frames)
    {
        for (const int obstacle_id : changed_obstacle_ids)
        {
            auto *obstacle = this->obstacles[obstacle_id];
            const auto position = obstacle->get_position(obstacle->get_is_static() ? 0 : frame);
            auto *replacement = new coal::CollisionObject(
                std::shared_ptr<coal::CollisionGeometry>(obstacle->clone_collision_geometry(frame)),
                position.rotation.toRotationMatrix(),
                position.pos);
            replacement->collisionGeometry()->computeLocalAABB();
            replacement->setUserData(
                this->frames_and_ids + frame + this->frame_count * obstacle_id);
            this->by_frame_spheres[frame][obstacle_id] = replacement;
        }
        auto *replacement_manager = new coal::DynamicAABBTreeArrayCollisionManager();
        replacement_manager->tree_init_level = 2;
        this->frame_broadphase_managers[frame] = replacement_manager;
        replacement_manager->registerObjects(this->by_frame_spheres[frame]);
        this->frame_broadphase_managers[frame]->setup();
    }

    const int total_obstacle_count = this->collision_object_count + this->robot_obstacle_all_joint_count;
    for (const int obstacle_id : changed_obstacle_ids)
    {
        auto *obstacle = this->obstacles[obstacle_id];
        for (int frame = 0; frame <= maximum_indexed_frame; ++frame)
        {
            const int first = frame;
            int last = frame;
            const auto base_position = obstacle->get_position(obstacle->get_is_static() ? 0 : frame);
            if (obstacle->get_is_static())
            {
                last = maximum_indexed_frame;
            }
            else
            {
                while (last + 1 <= maximum_indexed_frame)
                {
                    const auto next = obstacle->get_position(last + 1);
                    if (!same_coordinate(base_position, next) ||
                        obstacle->get_security_margin(first) != obstacle->get_security_margin(last + 1))
                    {
                        break;
                    }
                    ++last;
                }
            }
            auto *replacement = new coal::CollisionObject(
                std::shared_ptr<coal::CollisionGeometry>(obstacle->clone_collision_geometry(first)),
                base_position.rotation.toRotationMatrix(),
                base_position.pos);
            auto *slot = this->frames_and_ids_safe_interval +
                first * total_obstacle_count + obstacle_id;
            slot->first = {first, last};
            slot->second = obstacle_id;
            replacement->setUserData(slot);
            replacement->collisionGeometry()->computeLocalAABB();
            this->all_spheres.push_back(replacement);
            frame = last;
        }
        this->safe_interval_indexed_until[obstacle_id] = maximum_indexed_frame;
    }
    this->minimum_indexed_until_frame = maximum_indexed_frame;
    auto *replacement_safe_interval_manager = new coal::DynamicAABBTreeArrayCollisionManager();
    replacement_safe_interval_manager->tree_init_level = 2;
    this->broadphase_manager = replacement_safe_interval_manager;
    this->broadphase_manager->registerObjects(this->all_spheres);
    this->broadphase_manager->setup();

    const auto robot_joints = this->planned_robot.get_collision_object_for_robot_angles(
        updated_scene.start_configuration);
    for (int robot_joint_id = 0; robot_joint_id < this->collision_robot_links_count; ++robot_joint_id)
    {
        for (int obstacle_id = 0; obstacle_id < this->collision_object_count; ++obstacle_id)
        {
            this->collision_pairs.emplace_back(
                robot_joints[robot_joint_id + this->robot_joint_offset].collision_object.get(),
                this->obstacles[obstacle_id]->get_collision_object());
        }
    }
    auto replacement_scene = updated_scene;
    this->scene_task = std::move(replacement_scene);
    return report;
}
bool MDP::CollisionManager::check_collision(const std::vector<double> &robot_angles, float &time)

{
    int frame = (int)(time * (this->scene_task.fps));
    assert(frame<this->scene_task.frame_count);
    return MDP::ResultsWriter::get_instance().collision_check_wrapper(std::bind(&MDP::CollisionManager::check_collision_private, this, std::placeholders::_1, std::placeholders::_2), robot_angles, frame);
}

bool MDP::CollisionManager::check_collision_frame(const std::vector<double> &robot_angles, int &frame)
{
    return MDP::ResultsWriter::get_instance().collision_check_wrapper(std::bind(&MDP::CollisionManager::check_collision_private, this, std::placeholders::_1, std::placeholders::_2), robot_angles, frame);
}

bool MDP::CollisionManager::check_collision_frame_no_wrapper(std::vector<double> robot_angles, int frame)
{
    return this->check_collision_private(robot_angles, frame);
}

bool MDP::CollisionManager::check_collision_private(const std::vector<double> &robot_angles, int &frame)
{
    if (frame < 0 || frame > this->minimum_indexed_until_frame)
    {
        throw std::out_of_range("collision frame exceeds the indexed horizon");
    }
    std::vector<MDP::RobotObstacleFCL::JointCollisionObject> main_robot_collision_models = MDP::ResultsWriter::get_instance().forward_kinematics_collision_check_wrapper(std::bind(&MDP::RobotObstacleFCL::get_collision_object_for_robot_angles, &(this->planned_robot), std::placeholders::_1), robot_angles);

    std::vector<MDP::RobotObstacleFCL::JointCollisionObject> obstacle_joints;

    // bool is_collided = false;

    MDP::CollisionCallback collision_callback(&(this->collision_pairs), this->collision_object_count + this->robot_obstacle_all_joint_count, &this->obstacle_security_margins);
    for (int robot_joint_id = 0; robot_joint_id < this->collision_robot_links_count; robot_joint_id++)
    {
        MDP::ResultsWriter::get_instance().collision_check_broadphase_counter += 1;

        collision_callback.is_collided = false;
        coal::CollisionObject joint_coll_obj(main_robot_collision_models[robot_joint_id+this->robot_joint_offset].collision_object, main_robot_collision_models[robot_joint_id+this->robot_joint_offset].transform);
        joint_coll_obj.setUserData(new std::pair<int, int>(-1, robot_joint_id));
        // std::cout<<"_________________"<<std::endl;
        // std::cout<<frame<<std::endl;
        // std::cout<<&this->collision_pairs<<std::endl;
        // std::cout <<(robot_joint_id)<<std::endl;
        // std::cout<<joint_coll_obj.getTransform().getTranslation()<<" "<<((coal::Capsule*)(joint_coll_obj.collisionGeometry().get()))->radius<<" "<<((coal::Capsule*)(joint_coll_obj.collisionGeometry().get()))->halfLength<<std::endl;
        this->frame_broadphase_managers[frame]->collide(&joint_coll_obj, &collision_callback);
        delete (std::pair<int, int> *)joint_coll_obj.getUserData();

        if (collision_callback.is_collided)
        {
            MDP::ResultsWriter::get_instance().collision_count+=1;
            return true;
        }
    }

    return false;
}

std::vector<MDP::CollisionManager::robot_link_distance_to_obj> MDP::CollisionManager::get_distances(std::vector<double> robot_angles, float time, bool &is_collision)
{
    return MDP::ResultsWriter::get_instance().distance_check_wrapper(std::bind(&MDP::CollisionManager::get_distances_private, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), robot_angles, time, is_collision);
}

std::vector<MDP::CollisionManager::robot_link_distance_to_obj> MDP::CollisionManager::get_distances_private(std::vector<double> robot_angles, float time, bool &is_collision)
{
    is_collision = false;
    const int frame = static_cast<int>(time * this->scene_task.fps);
    if (frame < 0 || frame > this->minimum_indexed_until_frame)
    {
        throw std::out_of_range("distance frame exceeds the indexed horizon");
    }

    std::vector<MDP::CollisionManager::robot_link_distance_to_obj> results;

    // move robots and dynamic obstacles
    std::vector<MDP::RobotObstacleFCL::JointCollisionObject> main_robot_collision_models = MDP::ResultsWriter::get_instance().forward_kinematics_distance_check_wrapper(std::bind(&MDP::RobotObstacleFCL::get_collision_object_for_robot_angles, &(this->planned_robot), std::placeholders::_1), robot_angles);

    MDP::DistanceCallback distance_callback(&this->collision_pairs, this->collision_object_count + this->robot_obstacle_all_joint_count);
    for (int robot_joint_id = 0; robot_joint_id < this->collision_robot_links_count; robot_joint_id++)
    {
        distance_callback.init();
        float min_distance = std::numeric_limits<float>::max();
        std::vector<Eigen::Vector3d> robot_links_points;
        std::vector<Eigen::Vector3d> obj_points;

        coal::CollisionObject joint_coll_obj(main_robot_collision_models[robot_joint_id+this->robot_joint_offset].collision_object, main_robot_collision_models[robot_joint_id+this->robot_joint_offset].transform);
        joint_coll_obj.setUserData(new std::pair<int, int>(-1, robot_joint_id));
        this->frame_broadphase_managers[frame]->distance(&joint_coll_obj, &distance_callback);
        delete (std::pair<int, int> *)joint_coll_obj.getUserData();

        if (distance_callback.is_collided)
        {
            is_collision = true;
            return results;
        }
        results.emplace_back(distance_callback.dist, distance_callback.robot_links_points, distance_callback.obj_points);
    }

    // multirobot is not implemented.
    return results;
};

std::vector<std::pair<float, float>> MDP::CollisionManager::get_planned_robot_limits() const
{
    return this->planned_robot.get_limits();
}

int MDP::CollisionManager::get_goal_frame_low_bound()
{
    for (int frame = this->scene_task.frame_count - 1; frame >= 0; frame--)
    {
        std::vector<double> end_configuration_f(scene_task.end_configuration.begin(), scene_task.end_configuration.end());
        bool collision_result = this->check_collision_frame(end_configuration_f, frame); // TODO: conversion from frame to double and fropm double to frame may cause bugs
        if (collision_result)
        {
            if (frame == this->scene_task.frame_count - 1)
            {
                throw std::runtime_error("there is collision in goal state at the last frame!");
            }
            return (frame + 1);
        }
    }
    return 0;
}

bool MDP::CollisionManager::check_robot_selfcollision(std::vector<double> robot_angles)
{
    return false;
    // move robots and dynamic obstacles
    std::vector<MDP::RobotObstacleFCL::JointCollisionObject> main_robot_collision_models = MDP::ResultsWriter::get_instance().forward_kinematics_collision_check_wrapper(std::bind(&MDP::RobotObstacleFCL::get_collision_object_for_robot_angles, &(this->planned_robot), std::placeholders::_1), robot_angles);

    bool is_collided = false;
    int ind = 0;
    for (MDP::RobotObstacleFCL::JointCollisionObject &joint_mesh_and_transfrom : main_robot_collision_models) // for robot joint mesh
    {
        if (ind == main_robot_collision_models.size() - 2)
        {
            return false;
        }
        // don't check neighbour link
        for (int another_joint_ind = ind + 2; another_joint_ind < main_robot_collision_models.size(); another_joint_ind++)
        {
            MDP::RobotObstacleFCL::JointCollisionObject &another_joint_mesh_and_transfrom = main_robot_collision_models[another_joint_ind];

            // bool self_collision = this->collide_objects(joint_mesh_and_transfrom.collision_object.get(), joint_mesh_and_transfrom.transform,
            //                                             another_joint_mesh_and_transfrom.collision_object.get(), another_joint_mesh_and_transfrom.transform);
            // if (self_collision)
            // {

            //     return true;
            // }
        }
        ind++;
    }
}

bool MDP::CollisionManager::check_base_joints_collisiion(std::vector<double> robot_angles, int last_base_joint_ind)
{
    return false;
}

std::vector<std::pair<int, int>> MDP::CollisionManager::get_safe_intervals(const std::vector<double> &robot_angles)
{
    return this->get_safe_intervals(robot_angles, this->frame_count - 1);
}

std::vector<std::pair<int, int>> MDP::CollisionManager::get_safe_intervals(
    const std::vector<double> &robot_angles,
    int maximum_frame)
{
    if (maximum_frame < 0 || maximum_frame >= this->frame_count ||
        maximum_frame > this->minimum_indexed_until_frame)
    {
        throw std::out_of_range("safe-interval maximum frame is outside the scene");
    }
    return MDP::ResultsWriter::get_instance().safe_interval_wrapper(
        [this, maximum_frame](const std::vector<double> &angles) {
            return this->get_safe_intervals_private(angles, maximum_frame);
        },
        robot_angles);
}

std::vector<std::pair<int, int>> MDP::CollisionManager::get_safe_intervals_private(
    const std::vector<double> &robot_angles,
    int maximum_frame)
{
    std::vector<std::pair<int, int>> results;

    int start_safe_frame = 0;
    bool is_safe = false;

    std::vector<MDP::RobotObstacleFCL::JointCollisionObject> main_robot_collision_models = MDP::ResultsWriter::get_instance().forward_kinematics_collision_check_wrapper(std::bind(&MDP::RobotObstacleFCL::get_collision_object_for_robot_angles, &(this->planned_robot), std::placeholders::_1), robot_angles);

    MDP::SafeIntervalCollisionCallback collision_callback(
        &this->collision_pairs,
        this->collision_object_count + this->robot_obstacle_all_joint_count,
        &this->obstacle_security_margins,
        maximum_frame);
    for (int robot_joint_id = 0; robot_joint_id < this->collision_robot_links_count; robot_joint_id++)
    {
        MDP::ResultsWriter::get_instance().safe_interval_broadphase_counter+=1;
        coal::CollisionObject joint_coll_obj(main_robot_collision_models[robot_joint_id+this->robot_joint_offset].collision_object, main_robot_collision_models[robot_joint_id+this->robot_joint_offset].transform);
        std::pair<std::pair<int, int>, int> *joint_ind = new std::pair<std::pair<int, int>, int>(std::pair<int, int>(-1, -1), robot_joint_id);
        joint_coll_obj.setUserData(joint_ind);
        // std::cout<<"1"<<std::endl;
        this->broadphase_manager->collide(&joint_coll_obj, &collision_callback);
        delete joint_ind;
        // std::cout<<"2"<<std::endl;
    }

    std::vector<int> frames_in_collision(collision_callback.set_frames_in_collision.begin(), collision_callback.set_frames_in_collision.end());

    if (frames_in_collision.size() == 0)
    {
        MDP::ResultsWriter::get_instance().sum_of_safe_interval_frames += maximum_frame + 1;
        MDP::ResultsWriter::get_instance().number_of_safe_intervals += 1;

        results.emplace_back(0, maximum_frame);
        // std::cout<<"safe interval: "<<0<<" "<<scene_task.frame_count - 1<<std::endl;
        return results;
    }
    // sort and delete duplicates
    // std::sort(frames_in_collision.begin(), frames_in_collision.end());
    // frames_in_collision.erase(std::unique(frames_in_collision.begin(), frames_in_collision.end()), frames_in_collision.end());

    int last_collision_frame = -1;
    for (const int collision_frame : frames_in_collision)
    {
        if ((collision_frame - last_collision_frame) != 1)
        {
            MDP::ResultsWriter::get_instance().sum_of_safe_interval_frames+= collision_frame - 1 -( last_collision_frame + 1) + 1; 
            results.emplace_back(last_collision_frame + 1, collision_frame - 1);
        }

        last_collision_frame = collision_frame;
    }
    if (frames_in_collision.back() != maximum_frame)
    {
        MDP::ResultsWriter::get_instance().sum_of_safe_interval_frames +=
            maximum_frame - (frames_in_collision.back() + 1) + 1;

        results.emplace_back(frames_in_collision.back() + 1, maximum_frame);
    }

    // for (const auto &result : results)
    // {
    //     std::cout<<"safe interval: "<<result.first<<" "<<result.second<<std::endl;
    // }
    // std::cout<<"--------------------------------"<<std::endl;
    MDP::ResultsWriter::get_instance().number_of_safe_intervals += results.size();
    return results;
}

bool MDP::DistanceCallback::distance(coal::CollisionObject *o1, coal::CollisionObject *o2, coal::Scalar &dist)
{
    if (this->is_collided)
    {
        return this->is_collided;
    }
    // collide
    coal::DistanceRequest dis_req;
    coal::DistanceResult dis_res;
    coal::CollisionRequest col_req;
    col_req.security_margin = 0.000000000000000000001;
    coal::CollisionResult col_res;
    int *robot_joint_id;
    std::pair<int, int> *obstacle_frame_and_id1;
    std::pair<int, int> *obstacle_frame_and_id2;
    if (!((o1->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE && o2->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE) || (o1->getNodeType() == coal::NODE_TYPE::GEOM_SPHERE && o2->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE) || (o2->getNodeType() == coal::NODE_TYPE::GEOM_SPHERE && o1->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE)))
    {
        // std::cout<<"AABB"<<std::endl;
        // assert(false);
        coal::distance(o1, o2, dis_req, dis_res);
        dist = dis_res.min_distance;
        this->dist = dis_res.min_distance;
        this->is_collided = dis_res.min_distance < 0;
        this->robot_links_points.push_back(dis_res.nearest_points.at(0));
        this->obj_points.push_back(dis_res.nearest_points.at(1));
        return this->is_collided;
    }

    obstacle_frame_and_id1 = (std::pair<int, int> *)o1->getUserData();
    obstacle_frame_and_id2 = (std::pair<int, int> *)o2->getUserData();
    if (obstacle_frame_and_id1->first == -1)
    {
        this->collision_pairs->at(obstacle_frame_and_id1->second * this->collision_object_count + (obstacle_frame_and_id2->second))(o1->getTransform(), o2->getTransform(), col_req, col_res);
    }
    else if (obstacle_frame_and_id2->first == -1)
    {
        this->collision_pairs->at(obstacle_frame_and_id2->second * this->collision_object_count + (obstacle_frame_and_id1->second))(o2->getTransform(), o1->getTransform(), col_req, col_res);
    }
    else
    {
        assert(false);
    }

    dist = col_res.distance_lower_bound;

    this->is_collided = col_res.isCollision();
    this->dist = col_res.distance_lower_bound; // distance_lower_bound == distance for simple shapes. TODO: check that info
    this->robot_links_points.push_back(col_res.nearest_points.at(0));
    this->obj_points.push_back(col_res.nearest_points.at(1));

    return this->is_collided;
}

bool MDP::CollisionCallback::collide(coal::CollisionObject *o1, coal::CollisionObject *o2)
{
    MDP::ResultsWriter::get_instance().collision_check_narrowphase_counter+=1;
    if (this->is_collided)
    {
        return this->is_collided;
    }
    // collide
    coal::CollisionRequest col_req;
    col_req.security_margin = 0.000000000000000000001;
    coal::CollisionResult col_res;

    int *robot_joint_id;
    std::pair<int, int> *obstacle_frame_and_id1;
    std::pair<int, int> *obstacle_frame_and_id2;
    // if (!((o1->getNodeType() == coal::NODE_TYPE::GEOM_BOX && o2->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE) ||(o1->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE && o2->getNodeType() == coal::NODE_TYPE::GEOM_BOX) ||(o1->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE && o2->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE) ||(o1->getNodeType() == coal::NODE_TYPE::GEOM_SPHERE && o2->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE) || (o2->getNodeType() == coal::NODE_TYPE::GEOM_SPHERE && o1->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE)))
    // {
    //     // std::cout<<"AABB"<<std::endl;
    //     assert(false);
    //     coal::collide(o1, o2, col_req, col_res);
    //     return col_res.isCollision();
    // }
    obstacle_frame_and_id1 = (std::pair<int, int> *)o1->getUserData();
    obstacle_frame_and_id2 = (std::pair<int, int> *)o2->getUserData();
    if (obstacle_frame_and_id1->first == -1)
    {
        const int frame = obstacle_frame_and_id2->first;
        const int obstacle_id = obstacle_frame_and_id2->second;
        if (obstacle_id < static_cast<int>(this->security_margins->size()) && this->security_margins->at(obstacle_id).at(frame) > 0.0)
        {
            coal::collide(o1, o2, col_req, col_res);
        }
        else
        {
            this->collision_pairs->at(obstacle_frame_and_id1->second * this->collision_object_count + obstacle_id)(o1->getTransform(), o2->getTransform(), col_req, col_res);
        }
    }
    else if (obstacle_frame_and_id2->first == -1)
    {
        const int frame = obstacle_frame_and_id1->first;
        const int obstacle_id = obstacle_frame_and_id1->second;
        if (obstacle_id < static_cast<int>(this->security_margins->size()) && this->security_margins->at(obstacle_id).at(frame) > 0.0)
        {
            coal::collide(o1, o2, col_req, col_res);
        }
        else
        {
            this->collision_pairs->at(obstacle_frame_and_id2->second * this->collision_object_count + obstacle_id)(o2->getTransform(), o1->getTransform(), col_req, col_res);
        }
    }
    else
    {
        assert(false);
    }

    // std::cout<<"sphere"<<std::endl;
    // std::cout<<"robot_joint_id "<<*robot_joint_id<<std::endl;
    // std::cout<<"obstacle_frame_and_id "<<obstacle_frame_and_id->first <<" "<<obstacle_frame_and_id->second<<std::endl;
    // coal::collide(o1, o2, col_req, col_res);
    this->is_collided = col_res.isCollision();

    return this->is_collided;
}

bool MDP::SafeIntervalCollisionCallback::collide(coal::CollisionObject *o1, coal::CollisionObject *o2)
{
    MDP::ResultsWriter::get_instance().safe_interval_narrowphase_counter += 1;

    // collide
    coal::CollisionRequest col_req;
    col_req.security_margin = 0.000000000000000000001;
    coal::CollisionResult col_res;

    int *robot_joint_id;
    std::pair<std::pair<int, int> , int> *obstacle_frame_and_id1;
    std::pair<std::pair<int, int> , int> *obstacle_frame_and_id2;
    int frame = 0;
    int frame_low = 0;
    int frame_high = 0;
    // if (!((o1->getNodeType() == coal::NODE_TYPE::GEOM_BOX && o2->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE) ||(o1->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE && o2->getNodeType() == coal::NODE_TYPE::GEOM_BOX) ||(o1->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE && o2->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE) ||(o1->getNodeType() == coal::NODE_TYPE::GEOM_SPHERE && o2->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE) || (o2->getNodeType() == coal::NODE_TYPE::GEOM_SPHERE && o1->getNodeType() == coal::NODE_TYPE::GEOM_CAPSULE)))
    // {
    //     std::cout << o1->getNodeType() << std::endl;
    //     std::cout << o2->getNodeType() << std::endl;
    //     assert(false);
    //     coal::collide(o1, o2, col_req, col_res);
    //     return col_res.isCollision();
    // }
    obstacle_frame_and_id1 = (std::pair<std::pair<int, int> , int> *)o1->getUserData();
    obstacle_frame_and_id2 = (std::pair<std::pair<int, int> , int> *)o2->getUserData();

    if (obstacle_frame_and_id1->first.first == -1)
    {
        frame_low = obstacle_frame_and_id2->first.first;
        frame_high = std::min(
            obstacle_frame_and_id2->first.second, this->maximum_frame);
        if (frame_low > frame_high)
        {
            return false;
        }
        
        // Check if all frames in the interval [frame_low, frame_high] are already in collision
        bool all_frames_in_collision = true;
        for (int frame = frame_low; frame <= frame_high; frame++) {
            if (this->set_frames_in_collision.find(frame) == this->set_frames_in_collision.end()) {
                all_frames_in_collision = false;
                break;  // Early termination - found a frame not in collision
            }
        }
        if (all_frames_in_collision) {
            return false;
        }
        
        this->collision_pairs->at(obstacle_frame_and_id1->second * this->collision_object_count + (obstacle_frame_and_id2->second))(o1->getTransform(), o2->getTransform(), col_req, col_res);
    }
    else if (obstacle_frame_and_id2->first.first == -1)
    {
        frame_low = obstacle_frame_and_id1->first.first;
        frame_high = std::min(
            obstacle_frame_and_id1->first.second, this->maximum_frame);
        if (frame_low > frame_high)
        {
            return false;
        }
        
        // Check if all frames in the interval [frame_low, frame_high] are already in collision
        bool all_frames_in_collision = true;
        for (int frame = frame_low; frame <= frame_high; frame++) {
            if (this->set_frames_in_collision.find(frame) == this->set_frames_in_collision.end()) {
                all_frames_in_collision = false;
                break;  // Early termination - found a frame not in collision
            }
        }
        if (all_frames_in_collision) {
            return false;
        }
        
        this->collision_pairs->at(obstacle_frame_and_id2->second * this->collision_object_count + (obstacle_frame_and_id1->second))(o2->getTransform(), o1->getTransform(), col_req, col_res);
    
    }
    else
    {
        assert(false);
    }

    // std::cout<<"sphere"<<std::endl;
    // std::cout<<"robot_joint_id "<<*robot_joint_id<<std::endl;
    // std::cout<<"obstacle_frame_and_id "<<obstacle_frame_and_id->first <<" "<<obstacle_frame_and_id->second<<std::endl;
    coal::collide(o1, o2, col_req, col_res);
    bool is_collided = col_res.isCollision();

    // if is_collision, then add frame to vector
    if (is_collided)
    {
        // std::cout<<"collision!"<<std::endl;
        // std::cout<<"obstacle_frame_and_id1->first "<<obstacle_frame_and_id1->first<<std::endl;
        // std::cout<<"obstacle_frame_and_id1->second "<<obstacle_frame_and_id1->second<<std::endl;
        // std::cout<<"obstacle_frame_and_id2->first "<<obstacle_frame_and_id2->first<<std::endl;
        // std::cout<<"obstacle_frame_and_id2->second "<<obstacle_frame_and_id2->second<<std::endl;

        for (int frame = frame_low; frame <= frame_high; frame++)
        {
            this->set_frames_in_collision.insert(frame);
        }
    }

    return false; // return false, because if return true, than broadphase ends.
}

void MDP::CollisionManager::benchmark_broadphase()
{

    std::vector<MDP::ResultsWriter::Timer> timers;
    coal::BroadPhaseCollisionManager *original_broadphase_manager = this->broadphase_manager;
    std::vector<coal::BroadPhaseCollisionManager *> managers;
    // managers.push_back(new coal::NaiveCollisionManager());
    managers.push_back(new coal::SSaPCollisionManager());
    managers.push_back(new coal::SaPCollisionManager());
    managers.push_back(new coal::IntervalTreeCollisionManager());
    coal::Vec3s lower_limit, upper_limit;
    lower_limit[0] = -1;
    lower_limit[1] = -1;
    lower_limit[2] = -1;

    upper_limit[0] = -1;
    upper_limit[1] = -1;
    upper_limit[2] = -1;
    coal::SpatialHashingCollisionManager<>::computeBound(this->all_spheres, lower_limit, upper_limit);
    coal::Scalar cell_size =
        std::min(std::min((upper_limit[0] - lower_limit[0]) / 20,
                          (upper_limit[1] - lower_limit[1]) / 20),
                 (upper_limit[2] - lower_limit[2]) / 20);

    // cell_size = 0.1;
    managers.push_back(
        new coal::SpatialHashingCollisionManager<
            coal::detail::SparseHashTable<coal::AABB, coal::CollisionObject *, coal::detail::SpatialHash>>(
            cell_size, lower_limit, upper_limit));

    managers.push_back(
        new coal::SpatialHashingCollisionManager<
            coal::detail::SparseHashTable<coal::AABB, coal::CollisionObject *, coal::detail::SpatialHash>>(
            cell_size, lower_limit, upper_limit));

    managers.push_back(new coal::DynamicAABBTreeCollisionManager());
    managers.push_back(new coal::DynamicAABBTreeArrayCollisionManager());

    {
        coal::DynamicAABBTreeCollisionManager *m = new coal::DynamicAABBTreeCollisionManager();
        m->tree_init_level = 2;
        managers.push_back(m);
    }

    {
        coal::DynamicAABBTreeArrayCollisionManager *m =
            new coal::DynamicAABBTreeArrayCollisionManager();
        m->tree_init_level = 2;
        managers.push_back(m);
    }

    {
        coal::DynamicAABBTreeArrayCollisionManager *m =
            new coal::DynamicAABBTreeArrayCollisionManager();
        m->tree_init_level = 12;
        m->max_tree_nonbalanced_level = 0;

        managers.push_back(m);
    }
    {
        coal::DynamicAABBTreeArrayCollisionManager *m =
            new coal::DynamicAABBTreeArrayCollisionManager();
        m->tree_init_level = 3;
        managers.push_back(m);
    }
    {
        coal::DynamicAABBTreeArrayCollisionManager *m =
            new coal::DynamicAABBTreeArrayCollisionManager();
        m->tree_init_level = 4;
        managers.push_back(m);
    }
    {
        coal::DynamicAABBTreeArrayCollisionManager *m =
            new coal::DynamicAABBTreeArrayCollisionManager();
        m->tree_init_level = 11;
        m->max_tree_nonbalanced_level = 3;
        managers.push_back(m);
    }

    timers.resize(managers.size());

    for (size_t i = 0; i < managers.size(); ++i)
    {
        timers[i].reset();
        managers[i]->registerObjects(this->all_spheres);
        timers[i].pause_timer();
        std::cout << i << "manager register time: " << timers[i].get_elapsed_time_ns() << std::endl;
    }

    for (size_t i = 0; i < managers.size(); ++i)
    {
        timers[i].reset();
        managers[i]->setup();
        timers[i].pause_timer();
        std::cout << i << "manager setup time: " << timers[i].get_elapsed_time_ns() << std::endl;
    }
    std::random_device r;
    std::mt19937 gen(r());
    std::uniform_real_distribution<> probability_gen(0.0, 1.0);

    std::vector<std::pair<float, float>> robot_limits = this->get_planned_robot_limits();
    std::vector<double> robot_angles;
    std::vector<long long> results;
    results.resize(managers.size());
    for (int i = 0; i < results.size(); i++)
    {
        results[i] = 0;
    }

    // delete this->broadphase_manager;

    robot_angles.resize(6);
    long long naive_si_time = 0;
    for (int j = 0; j < 1000; j++)
    {
        std::cout << "------------------------------" << j<< std::endl;
        for (int i = 0; i < 6; i++)
        {
            robot_angles[i] = (probability_gen(gen) * (robot_limits[i].second - robot_limits[i].first) + robot_limits[i].first);
            std::cout << robot_angles[i] << ", ";
        }
        std::cout << std::endl;

        // MDP::ResultsWriter::Timer rt;
        // rt.reset();
        // std::vector<std::pair<int, int>> res = this->get_safe_intervals_naive(robot_angles);
        // rt.pause_timer();
        // std::cout << "real interval: " << rt.get_elapsed_time_ns() << " " << res.size() << std::endl;
        // for (std::pair<int, int> r : res)
        // {
        //     std::cout << "[" << r.first << " " << r.second << "]  ";
        // }
        // std::cout << std::endl;
        // rt.reset();
        // res = this->get_safe_intervals_naive_fastcc(robot_angles);
        // rt.pause_timer();
        // naive_si_time+=rt.get_elapsed_time_ns();
        // std::cout << "real interval: " << rt.get_elapsed_time_ns() << " " << res.size() << std::endl;
        // for (std::pair<int, int> r : res)
        // {
        //     std::cout << "[" << r.first << " " << r.second << "]  ";
        // }
        // std::cout << std::endl;

        // rt.reset();
        // res = this->get_safe_intervals_naive_fastcc_oop(robot_angles);
        // rt.pause_timer();
        // std::cout << "real interval: " << rt.get_elapsed_time_ns() << " " << res.size() << std::endl;
        // for (std::pair<int, int> r : res)
        // {
        //     std::cout << "[" << r.first << " " << r.second << "]  ";
        // }
        // std::cout << std::endl;
        for (size_t i = 0; i < managers.size(); i++)
        {
            timers[i].reset();
            this->broadphase_manager = managers[i];
            std::vector<std::pair<int, int>> ints = this->get_safe_intervals(robot_angles);
            timers[i].pause_timer();
            results[i] += timers[i].get_elapsed_time_ns();
            std::cout << i << "manager collide time: " << timers[i].get_elapsed_time_ns() << " safe_interval_size: " << ints.size() << std::endl;
            for (std::pair<int, int> r : ints)
            {
                std::cout << "[" << r.first << " " << r.second << "]  ";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout <<  "naive_si_time: " << naive_si_time << std::endl;

    for (int i = 0; i < results.size(); i++)
    {
        std::cout << i << "manager collide time: " << results[i] << std::endl;
    }
}


std::vector<MDP::RobotObstacleFCL> MDP::CollisionManager::get_robot_obstacles() const
{
    return this->robot_obstacles;
}

std::vector<MDP::ObjectObstacleFCL *> MDP::CollisionManager::get_obstacles() const
{
    return this->obstacles;
}

MDP::ConfigReader::SceneTask MDP::CollisionManager::get_scene_task() const
{
    return this->scene_task;
}

MDP::RobotObstacleFCL MDP::CollisionManager::get_planned_robot() const{
    return this->planned_robot;
}
