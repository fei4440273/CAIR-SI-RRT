#include <nanoflann.hpp>
#include "Vertex.hpp"
#include "Tree.hpp"
#include "TreeRepair.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace
{
MDP::MSIRRT::FrameRange edge_frame_range(const MDP::MSIRRT::Vertex *vertex)
{
    const double first = std::min({
        vertex->parent->arrival_time,
        vertex->departure_from_parent_time,
        vertex->arrival_time});
    const double last = std::max({
        vertex->parent->arrival_time,
        vertex->departure_from_parent_time,
        vertex->arrival_time});
    return {
        static_cast<int>(std::floor(first)),
        static_cast<int>(std::ceil(last))};
}
} // namespace

MDP::MSIRRT::Tree::Tree(const std::string &tree_name_, size_t tree_idx_, const int& dimensionality) : tree_name(tree_name_), tree_idx(tree_idx_),
                                                                           kd_tree(dimensionality, *this, nanoflann::KDTreeSingleIndexAdaptorParams(25)) {};

MDP::MSIRRT::Tree::Tree(const int& dimensionality) : tree_name(""), tree_idx(-1), kd_tree(dimensionality, *this, nanoflann::KDTreeSingleIndexAdaptorParams(25)) {};

MDP::MSIRRT::Tree::~Tree() {

    for(int i=0;i< this->array_of_vertices.size();i++){
        delete array_of_vertices[i];
    }
    array_of_vertices.clear();
}; // destructor

void MDP::MSIRRT::Tree::add_vertex(const MDP::MSIRRT::Vertex::VertexCoordType &q_new_coords, std::pair<int, int> q_new_safe_interval , MDP::MSIRRT::Vertex *q_parent,  double departure_time, double arrival_time)
{   
    MDP::MSIRRT::Vertex* q_new = new MDP::MSIRRT::Vertex(q_new_coords, q_new_safe_interval);
    q_new->tree_id = this->tree_idx;
    q_new->parent = q_parent;
    q_new->arrival_time = arrival_time;
    q_new->departure_from_parent_time = departure_time;
    this->array_of_vertices.push_back(q_new);

    size_t N{array_of_vertices.size() - 1};

    q_new->vertex_id = N;
    q_new->active = true;
    q_new->prediction_version = this->prediction_version_;
    ++this->active_vertex_count_;

    // this->array_of_vertices.back().ID_in_array = N;
    this->kd_tree.addPoints(N, N);

    if (q_parent != nullptr)
    {
        q_parent->children.push_back(this->array_of_vertices.back());
        this->temporal_dependencies_.add_edge(q_new->vertex_id, edge_frame_range(q_new));
    }
    this->temporal_dependencies_.add_vertex(
        q_new->vertex_id,
        MDP::MSIRRT::FrameRange(q_new_safe_interval.first, q_new_safe_interval.second));
}

void MDP::MSIRRT::Tree::add_vertex(const std::vector<double> &q_new_coords, std::pair<int, int> q_new_safe_interval, MDP::MSIRRT::Vertex *q_parent, double departure_time, double arrival_time){
    this->add_vertex(MDP::MSIRRT::Vertex::VertexCoordType(q_new_coords.data()), q_new_safe_interval, q_parent, departure_time, arrival_time);
}

std::vector<MDP::MSIRRT::Vertex *> MDP::MSIRRT::Tree::deactivate_subtree(
    MDP::MSIRRT::Vertex *subtree_root,
    std::size_t prediction_version)
{
    std::vector<MDP::MSIRRT::Vertex *> deactivated;
    if (subtree_root == nullptr || !subtree_root->active)
    {
        return deactivated;
    }

    if (subtree_root->parent != nullptr)
    {
        auto &siblings = subtree_root->parent->children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), subtree_root), siblings.end());
    }

    std::vector<MDP::MSIRRT::Vertex *> pending{subtree_root};
    while (!pending.empty())
    {
        MDP::MSIRRT::Vertex *vertex = pending.back();
        pending.pop_back();
        if (!vertex->active)
        {
            continue;
        }
        pending.insert(pending.end(), vertex->children.begin(), vertex->children.end());
        this->kd_tree.removePoint(vertex->vertex_id);
        this->temporal_dependencies_.remove_vertex(vertex->vertex_id);
        this->temporal_dependencies_.remove_edge(vertex->vertex_id);
        vertex->active = false;
        vertex->invalidated_prediction_version = prediction_version;
        --this->active_vertex_count_;
        deactivated.push_back(vertex);
    }
    return deactivated;
}

std::size_t MDP::MSIRRT::Tree::active_vertex_count() const
{
    return this->active_vertex_count_;
}

void MDP::MSIRRT::Tree::set_prediction_version(std::size_t prediction_version)
{
    this->prediction_version_ = prediction_version;
}

void MDP::MSIRRT::Tree::update_vertex_interval(
    MDP::MSIRRT::Vertex *vertex,
    std::pair<int, int> safe_interval,
    std::size_t prediction_version)
{
    if (vertex == nullptr || !vertex->active)
    {
        return;
    }
    vertex->safe_interval = safe_interval;
    vertex->prediction_version = prediction_version;
    this->temporal_dependencies_.add_vertex(
        vertex->vertex_id,
        MDP::MSIRRT::FrameRange(safe_interval.first, safe_interval.second));
}

MDP::MSIRRT::AffectedDependencies MDP::MSIRRT::Tree::affected_by(
    const std::vector<MDP::MSIRRT::FrameRange> &changed_windows) const
{
    return this->temporal_dependencies_.query(changed_windows);
}

MDP::MSIRRT::TreeRerootReport MDP::MSIRRT::Tree::consume_outside_subtree(
    MDP::MSIRRT::Vertex *new_root,
    std::size_t active_vertices_before,
    std::size_t prediction_version,
    bool root_is_new)
{
    std::unordered_set<std::size_t> preserved;
    std::vector<MDP::MSIRRT::Vertex *> pending{new_root};
    while (!pending.empty())
    {
        MDP::MSIRRT::Vertex *vertex = pending.back();
        pending.pop_back();
        if (!vertex->active || !preserved.insert(vertex->vertex_id).second)
        {
            continue;
        }
        pending.insert(pending.end(), vertex->children.begin(), vertex->children.end());
    }

    for (MDP::MSIRRT::Vertex *vertex : this->array_of_vertices)
    {
        if (vertex->active && preserved.count(vertex->vertex_id) == 0)
        {
            this->deactivate_subtree(vertex, prediction_version);
        }
    }

    MDP::MSIRRT::TreeRerootReport report;
    report.root = new_root;
    report.active_vertices_before = active_vertices_before;
    const std::size_t reused_old_vertices = preserved.size() - (root_is_new ? 1 : 0);
    report.reused_vertices = reused_old_vertices;
    report.consumed_vertices = active_vertices_before > reused_old_vertices
        ? active_vertices_before - reused_old_vertices
        : 0;
    report.invariants_hold = MDP::MSIRRT::check_tree_invariants(*this);
    return report;
}

MDP::MSIRRT::TreeRerootReport MDP::MSIRRT::Tree::reroot_at_vertex(
    MDP::MSIRRT::Vertex *new_root,
    std::pair<int, int> safe_interval,
    double arrival_time,
    std::size_t prediction_version)
{
    if (new_root == nullptr || !new_root->active || new_root->tree_id != static_cast<int>(this->tree_idx))
    {
        throw std::invalid_argument("new root must be an active vertex in this tree");
    }
    if (arrival_time < safe_interval.first || arrival_time > safe_interval.second)
    {
        throw std::invalid_argument("new root arrival time is outside its safe interval");
    }
    const std::size_t active_before = this->active_vertex_count();
    if (new_root->parent != nullptr)
    {
        auto &siblings = new_root->parent->children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), new_root), siblings.end());
    }
    this->temporal_dependencies_.remove_edge(new_root->vertex_id);
    new_root->parent = nullptr;
    new_root->departure_from_parent_time = -1.0;
    new_root->arrival_time = arrival_time;
    this->update_vertex_interval(new_root, safe_interval, prediction_version);
    for (MDP::MSIRRT::Vertex *child : new_root->children)
    {
        if (child->active)
        {
            this->temporal_dependencies_.add_edge(child->vertex_id, edge_frame_range(child));
        }
    }
    return this->consume_outside_subtree(new_root, active_before, prediction_version, false);
}

MDP::MSIRRT::TreeRerootReport MDP::MSIRRT::Tree::reroot_on_edge(
    MDP::MSIRRT::Vertex *edge_child,
    const MDP::MSIRRT::Vertex::VertexCoordType &root_coords,
    std::pair<int, int> safe_interval,
    double arrival_time,
    std::size_t prediction_version)
{
    if (edge_child == nullptr || !edge_child->active || edge_child->parent == nullptr ||
        edge_child->tree_id != static_cast<int>(this->tree_idx))
    {
        throw std::invalid_argument("edge child must be an active non-root vertex in this tree");
    }
    if (arrival_time < safe_interval.first || arrival_time > safe_interval.second ||
        arrival_time > edge_child->arrival_time)
    {
        throw std::invalid_argument("edge reroot time is invalid");
    }
    const std::size_t active_before = this->active_vertex_count();
    MDP::MSIRRT::Vertex *old_parent = edge_child->parent;
    auto &siblings = old_parent->children;
    siblings.erase(std::remove(siblings.begin(), siblings.end(), edge_child), siblings.end());
    this->temporal_dependencies_.remove_edge(edge_child->vertex_id);

    this->set_prediction_version(prediction_version);
    this->add_vertex(root_coords, safe_interval, nullptr, -1.0, arrival_time);
    MDP::MSIRRT::Vertex *new_root = this->array_of_vertices.back();
    edge_child->parent = new_root;
    edge_child->departure_from_parent_time = arrival_time;
    new_root->children.push_back(edge_child);
    this->temporal_dependencies_.add_edge(edge_child->vertex_id, edge_frame_range(edge_child));
    return this->consume_outside_subtree(new_root, active_before, prediction_version, true);
}

// void MDP::MSIRRT::Tree::delete_vertex(int vertex_id)
// {
//     this->kd_tree.removePoint(vertex_id);
// }
