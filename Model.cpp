#include "Model.hpp"

#include <ClpSimplex.hpp>

#include <vector>

tsplp::Model::Model(size_t numberOfBinaryVariables)
    : m_spSimplexModel{std::make_unique<ClpSimplex>()}
{
    std::vector<double> lowerBounds(0.0, numberOfBinaryVariables);
    std::vector<double> upperBounds(1.0, numberOfBinaryVariables);

    m_spSimplexModel->addColumns(numberOfBinaryVariables, lowerBounds.data(), upperBounds.data(), nullptr, nullptr, nullptr, nullptr);
}
