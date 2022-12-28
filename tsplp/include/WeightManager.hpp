#pragma once

#include "DependencyHelpers.hpp"

#include <xtensor/xtensor.hpp>

#include <memory>
#include <vector>

namespace tsplp
{
class WeightManager
{
private:
    xt::xtensor<int, 2> m_weights;
    xt::xtensor<size_t, 1> m_startPositions;
    xt::xtensor<size_t, 1> m_endPositions;
    std::vector<size_t> m_toOriginal;
    std::unique_ptr<DependencyGraph> m_spDependencies;
    size_t m_originalN;

    size_t ToOriginal(size_t i) const;

public:
    WeightManager(
        xt::xtensor<int, 2> weights, xt::xtensor<size_t, 1> startPositions,
        xt::xtensor<size_t, 1> endPositions);

    const auto& W() const { return m_weights; }
    const auto& StartPositions() const { return m_startPositions; }
    const auto& EndPositions() const { return m_endPositions; }
    auto A() const { return m_startPositions.shape(0); };
    auto N() const { return m_weights.shape(0); }
    const auto& Dependencies() const { return *m_spDependencies; }

    std::vector<std::vector<size_t>> TransformPathsBack(
        std::vector<std::vector<size_t>> paths) const;

    xt::xtensor<double, 3> TransformTensorBack(const xt::xtensor<double, 3>& tensor) const;
};
}
