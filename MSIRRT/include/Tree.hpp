#pragma once

#include <nanoflann.hpp>
#include "RepairTypes.hpp"
#include "Vertex.hpp"

#define TREE_DIMENSIONALITY 6

namespace MDP::MSIRRT
{
    struct TreeRerootReport
    {
        MDP::MSIRRT::Vertex *root = nullptr;
        std::size_t active_vertices_before = 0;
        std::size_t reused_vertices = 0;
        std::size_t consumed_vertices = 0;
        bool invariants_hold = false;
    };

    struct Tree;
    typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Simple_Adaptor<double, MDP::MSIRRT::Tree>, MDP::MSIRRT::Tree> KdTree; 

    struct Tree
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Tree(const int& dimensionality);
        Tree(const std::string &tree_name_, size_t tree_idx_, const int& dimensionality);
        ~Tree();                                     // destructor
        Tree(const Tree &other) = delete;            // copy constructor
        Tree(Tree &&other) = default;                // move constructor
        Tree &operator=(const Tree &other) = delete; // copy assignment
        Tree &operator=(Tree &&other) = default;     // move assignment

        std::string tree_name;
        size_t tree_idx;

        std::vector<MDP::MSIRRT::Vertex*> array_of_vertices; //TODO: Maybe move to private?
        MDP::MSIRRT::KdTree kd_tree;

        MDP::MSIRRT::Vertex &getNearestState(const MDP::MSIRRT::Vertex q);

        void add_vertex(const MDP::MSIRRT::Vertex::VertexCoordType &q_new_coords, std::pair<int, int> q_new_safe_interval, MDP::MSIRRT::Vertex *q_parent, double departure_time, double arrival_time);
        void add_vertex(const std::vector<double> &q_new_coords, std::pair<int, int> q_new_safe_interval, MDP::MSIRRT::Vertex *q_parent, double departure_time, double arrival_time);

        std::vector<MDP::MSIRRT::Vertex *> deactivate_subtree(MDP::MSIRRT::Vertex *subtree_root, std::size_t prediction_version);
        std::size_t active_vertex_count() const;
        void set_prediction_version(std::size_t prediction_version);
        void update_vertex_interval(MDP::MSIRRT::Vertex *vertex, std::pair<int, int> safe_interval, std::size_t prediction_version);
        MDP::MSIRRT::AffectedDependencies affected_by(const std::vector<MDP::MSIRRT::FrameRange> &changed_windows) const;
        TreeRerootReport reroot_at_vertex(
            MDP::MSIRRT::Vertex *new_root,
            std::pair<int, int> safe_interval,
            double arrival_time,
            std::size_t prediction_version);
        TreeRerootReport reroot_on_edge(
            MDP::MSIRRT::Vertex *edge_child,
            const MDP::MSIRRT::Vertex::VertexCoordType &root_coords,
            std::pair<int, int> safe_interval,
            double arrival_time,
            std::size_t prediction_version);

        // void delete_vertex(int vertex_id);

        template <class BBOX>
        bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
        inline size_t kdtree_get_point_count() const { return array_of_vertices.size(); }
        inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return array_of_vertices[idx]->coords[dim]; }

    private:
        std::size_t active_vertex_count_ = 0;
        std::size_t prediction_version_ = 0;
        MDP::MSIRRT::TemporalDependencyIndex temporal_dependencies_;
        TreeRerootReport consume_outside_subtree(
            MDP::MSIRRT::Vertex *new_root,
            std::size_t active_vertices_before,
            std::size_t prediction_version,
            bool root_is_new);
    };
}
