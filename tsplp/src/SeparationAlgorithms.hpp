#pragma once

#include <optional>
#include <vector>
#include <xtensor/xtensor.hpp>

namespace tsplp
{
    class LinearConstraint;
    class Variable;
    class WeightManager;
}

namespace tsplp::graph
{
    class Separator
    {
    private:
        const xt::xtensor<Variable, 3>& m_variables;
        const WeightManager& m_weightManager;

    public:
        Separator(const xt::xtensor<Variable, 3>& variables, const WeightManager& weightManager);

        std::optional<LinearConstraint> Ucut() const;

        std::vector<LinearConstraint> Pi() const;
        std::vector<LinearConstraint> Sigma() const;
    };
}