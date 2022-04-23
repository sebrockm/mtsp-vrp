#pragma once

#include "DependencyHelpers.hpp"
#include "Variable.hpp"

#include <boost/graph/adjacency_matrix.hpp>

#include <xtensor/xtensor.hpp>

namespace tsplp::graph
{
    using PiSigmaGraphTraits = boost::adjacency_matrix_traits<boost::directedS>;
    using PiSigmaVertex = PiSigmaGraphTraits::vertex_descriptor;
    using PiSigmaEdge = PiSigmaGraphTraits::edge_descriptor;
    using PiSigmaVertexProperties =
        boost::property<boost::vertex_predecessor_t, PiSigmaGraphTraits::edge_descriptor,
        boost::property<boost::vertex_color_t, boost::default_color_type,
        boost::property<boost::vertex_distance_t, double>>>;
    using PiSigmaEdgeProperties = boost::property<boost::edge_residual_capacity_t, double>;
    using PiSigmaSupportGraphImpl = boost::adjacency_matrix<boost::directedS, PiSigmaVertexProperties, PiSigmaEdgeProperties>;

    class PiSigmaSupportGraph
    {
    private:
        PiSigmaSupportGraphImpl m_graph;
        const xt::xtensor<Variable, 3>& m_variables;
        const DependencyGraph& m_dependencies;
        const Model& m_model;

    public:
        PiSigmaSupportGraph(const xt::xtensor<Variable, 3>& variables, const DependencyGraph& dependencies, const Model& model);

    public:
        enum class ConstraintType
        {
            Pi, Sigma, PiSigma
        };

        std::pair<double, std::vector<std::pair<PiSigmaVertex, PiSigmaVertex>>> FindMinCut(PiSigmaVertex s, PiSigmaVertex t, ConstraintType x);
    };
}
