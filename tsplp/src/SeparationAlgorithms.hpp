#pragma once

#include <xtensor/xtensor.hpp>

#include <optional>

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
    Separator(
        const xt::xtensor<Variable, 3>& variables, const WeightManager& weightManager,
        const Model& model);
    ~Separator() noexcept;

    Separator(const Separator&) = delete;
    Separator(Separator&&) = delete;
    Separator& operator=(const Separator&) = delete;
    Separator& operator=(Separator&&) = delete;

    [[nodiscard]] std::optional<LinearConstraint> Ucut() const;

    [[nodiscard]] std::optional<LinearConstraint> Pi() const;
    [[nodiscard]] std::optional<LinearConstraint> Sigma() const;
    [[nodiscard]] std::optional<LinearConstraint> PiSigma() const;

    [[nodiscard]] std::vector<LinearConstraint> TwoMatching() const;
};
}
