#pragma once

#include <optional>
#include <xtensor/xtensor.hpp>

namespace tsplp
{
    class LinearConstraint;
    class Variable;
}

namespace tsplp::graph
{
    class Separator
    {
    private:
        const xt::xtensor<Variable, 3>& m_variables;

    public:
        Separator(const xt::xtensor<Variable, 3>& variables);

        std::optional<LinearConstraint> Ucut() const;
    };
}