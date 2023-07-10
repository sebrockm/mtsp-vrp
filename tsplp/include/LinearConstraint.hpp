#pragma once

#include "Variable.hpp"

#include <vector>

namespace tsplp
{
class LinearVariableComposition;
class Model;

class LinearConstraint
{
    friend LinearConstraint operator<=(
        LinearVariableComposition lhs, LinearVariableComposition rhs);
    friend LinearConstraint operator>=(
        LinearVariableComposition lhs, LinearVariableComposition rhs);
    friend LinearConstraint operator==(
        LinearVariableComposition lhs, LinearVariableComposition rhs);

private:
    std::unordered_map<size_t, double> m_variableIdCoefficientMap;
    double m_upperBound = 0.0;
    double m_lowerBound = 0.0;

private:
    explicit LinearConstraint(LinearVariableComposition&& convertee);

public:
    [[nodiscard]] double GetUpperBound() const { return m_upperBound; }
    [[nodiscard]] double GetLowerBound() const { return m_lowerBound; }
    [[nodiscard]] auto const& GetVariableIdCoefficientMap() const
    {
        return m_variableIdCoefficientMap;
    }

    [[nodiscard]] bool Evaluate(const Model& model, double tolerance = 1.e-10) const;
};

[[nodiscard]] LinearConstraint operator<=(
    LinearVariableComposition lhs, LinearVariableComposition rhs);
[[nodiscard]] LinearConstraint operator>=(
    LinearVariableComposition lhs, LinearVariableComposition rhs);
[[nodiscard]] LinearConstraint operator==(
    LinearVariableComposition lhs, LinearVariableComposition rhs);
}
