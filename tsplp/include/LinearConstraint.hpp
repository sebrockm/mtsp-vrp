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
    std::vector<Variable> m_variables;
    std::vector<double> m_coefficients;
    double m_upperBound = 0.0;
    double m_lowerBound = 0.0;

private:
    LinearConstraint(LinearVariableComposition&& convertee);

public:
    double GetUpperBound() const { return m_upperBound; }
    double GetLowerBound() const { return m_lowerBound; }
    auto const& GetCoefficients() const { return m_coefficients; }
    auto const& GetVariables() const { return m_variables; }

    bool Evaluate(const Model& model, double tolerance = 1.e-10) const;
};

LinearConstraint operator<=(LinearVariableComposition lhs, LinearVariableComposition rhs);
LinearConstraint operator>=(LinearVariableComposition lhs, LinearVariableComposition rhs);
LinearConstraint operator==(LinearVariableComposition lhs, LinearVariableComposition rhs);
}
