#pragma once

#include "Variable.hpp"

#include <vector>

namespace tsplp
{
class Model;

class LinearVariableComposition
{
    friend LinearVariableComposition operator*(double factor, LinearVariableComposition linearComp);

    friend LinearVariableComposition operator+(
        LinearVariableComposition lhs, LinearVariableComposition rhs);
    friend LinearVariableComposition operator+(LinearVariableComposition lhs, double rhs);

    friend LinearVariableComposition& operator+=(
        LinearVariableComposition& lhs, LinearVariableComposition const& rhs);

    friend LinearVariableComposition operator-(LinearVariableComposition operand);
    friend LinearVariableComposition operator-(
        LinearVariableComposition lhs, LinearVariableComposition rhs);

    friend class LinearConstraint;

private:
    std::vector<Variable> m_variables;
    std::vector<double> m_coefficients;
    double m_constant = 0;

public:
    LinearVariableComposition() = default;
    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    LinearVariableComposition(double constant);
    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    LinearVariableComposition(const Variable& variable);

    [[nodiscard]] auto const& GetVariables() const { return m_variables; }
    [[nodiscard]] auto const& GetCoefficients() const { return m_coefficients; }
    [[nodiscard]] double GetConstant() const { return m_constant; }

    [[nodiscard]] double Evaluate(const Model& model) const;
};

[[nodiscard]] LinearVariableComposition operator*(
    double factor, LinearVariableComposition linearComp);

[[nodiscard]] LinearVariableComposition operator+(
    LinearVariableComposition lhs, LinearVariableComposition rhs);
[[nodiscard]] LinearVariableComposition operator+(LinearVariableComposition lhs, double rhs);

LinearVariableComposition& operator+=(
    LinearVariableComposition& lhs, LinearVariableComposition const& rhs);

[[nodiscard]] LinearVariableComposition operator-(LinearVariableComposition operand);
[[nodiscard]] LinearVariableComposition operator-(
    LinearVariableComposition lhs, LinearVariableComposition rhs);
}
