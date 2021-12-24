#pragma once

#include <xtensor/xtensor.hpp>

#include <unordered_map>
#include <vector>

namespace tsplp
{
    class WeightManager
    {
    private:
        xt::xtensor<double, 2> m_weights;
        xt::xtensor<int, 1> m_startPositions;
        xt::xtensor<int, 1> m_endPositions;
        std::unordered_map<int, int> m_toOriginal;

    public:
        WeightManager(xt::xtensor<int, 2> weights, xt::xtensor<int, 1> startPositions, xt::xtensor<int, 1> endPositions);

        const auto& W() const { return m_weights; }
        const auto& StartPositions() const { return m_startPositions; }
        const auto& EndPositions() const { return m_endPositions; }
        const auto A() const { return m_startPositions.shape(0); };
        const auto N() const { return m_weights.shape(0); }

        std::vector<std::vector<int>> TransformPathsBack(std::vector<std::vector<int>> paths) const;
    };
}