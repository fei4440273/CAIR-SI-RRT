#pragma once

#include <vector>
#include <cstddef>
#include <limits>
#include <Eigen/Core>

namespace MDP::MSIRRT
{
    struct Vertex
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        typedef Eigen::Matrix<double, 6, 1> VertexCoordType; // TODO: Remove hardcode for 6 joints
        
        // Конструктор с параметрами (инициализация конфигурации)
        Vertex(const std::vector<double> &coords_, std::pair<int, int> safe_interval_);
        Vertex(const VertexCoordType &coords_, std::pair<int, int> safe_interval_);
        ~Vertex();                                        // destructor
        Vertex(const Vertex &other) = default;            // copy constructor
        Vertex(Vertex &&other) = delete;                  // move constructor
        Vertex &operator=(const Vertex &other) = default; // copy assignment
        Vertex &operator=(Vertex &&other) = delete;       // move assignment

        const VertexCoordType coords;      // Конфигурация
        std::pair<int, int> safe_interval; // Вектор безопасных интервалов [t_low, t_high]
        Vertex *parent = nullptr;          // указатель на родителя
        double arrival_time = -1;
        double before_rewiring_arrival_time = -1;
        double departure_from_parent_time = -1;
        std::vector<Vertex *> children{};
        int tree_id = -1;
        std::size_t vertex_id = std::numeric_limits<std::size_t>::max();
        bool active = true;
        std::size_t prediction_version = 0;
        std::size_t invalidated_prediction_version = std::numeric_limits<std::size_t>::max();
    };
}
