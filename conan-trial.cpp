#include <boost/graph/adjacency_matrix.hpp>
#include <ClpSimplex.hpp>

#include <iostream>

int main()
{
    ClpSimplex model;
    int result = model.readMps("C:\\Users\\sebas\\Downloads\\adlittle.mps");

    boost::adjacency_matrix<> graph{ 5 };
    for (auto u : graph.m_vertex_set)
        for (auto v : graph.m_vertex_set)
            if (u != v)
                boost::add_edge(u, v, graph);

    std::cout << "K5 edges: " << graph.m_num_edges << std::endl;
}
