#pragma once

#include <xtensor/xtensor.hpp>

#include <unordered_map>
#include <vector>

namespace tsplp
{
    class WeightManager
    {
    private:
        xt::xtensor<int, 2> m_weights;
        xt::xtensor<size_t, 1> m_startPositions;
        xt::xtensor<size_t, 1> m_endPositions;
        std::unordered_map<size_t, size_t> m_toOriginal;
        bool m_hasDependencies;

    public:
        WeightManager(xt::xtensor<int, 2> weights, xt::xtensor<size_t, 1> startPositions, xt::xtensor<size_t, 1> endPositions);

        const auto& W() const { return m_weights; }
        const auto& StartPositions() const { return m_startPositions; }
        const auto& EndPositions() const { return m_endPositions; }
        const auto A() const { return m_startPositions.shape(0); };
        const auto N() const { return m_weights.shape(0); }
        bool HasDependencies() const { return m_hasDependencies; }

        std::vector<std::vector<size_t>> TransformPathsBack(std::vector<std::vector<size_t>> paths) const;
    };
}