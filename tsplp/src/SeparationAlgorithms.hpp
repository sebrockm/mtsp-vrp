#pragma once

#include <optional>

#include <xtensor/xtensor.hpp>

namespace tsplp
{
    class LinearConstraint;
    class Variable;
    class WeightManager;
}

namespace tsplp::graph
{
    class PiSigmaSupportGraph;

    class Separator
    {
    private:
        const xt::xtensor<Variable, 3>& m_variables;
        const WeightManager& m_weightManager;
        std::unique_ptr<PiSigmaSupportGraph> m_spSupportGraph;

    public:
        Separator(const xt::xtensor<Variable, 3>& variables, const WeightManager& weightManager);
        ~Separator();

        std::optional<LinearConstraint> Ucut() const;

        std::optional<LinearConstraint> Pi() const;
        std::optional<LinearConstraint> Sigma() const;
        std::optional<LinearConstraint> PiSigma() const;
    };
}
