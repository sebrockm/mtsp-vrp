#pragma once

#include <optional>

#include <xtensor/xtensor.hpp>

namespace tsplp
{
    class LinearConstraint;
    class Model;
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
        const Model& m_model;
        std::unique_ptr<PiSigmaSupportGraph> m_spSupportGraph;

    public:
        Separator(const xt::xtensor<Variable, 3>& variables, const WeightManager& weightManager, const Model& model);
        ~Separator();

        std::optional<LinearConstraint> Ucut() const;

        std::optional<LinearConstraint> Pi() const;
        std::optional<LinearConstraint> Sigma() const;
        std::optional<LinearConstraint> PiSigma() const;
    };
}
