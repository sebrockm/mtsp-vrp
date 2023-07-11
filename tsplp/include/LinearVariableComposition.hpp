#pragma once

#include "Variable.hpp"

#include <unordered_map>

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
    std::unordered_map<size_t, double> m_variableIdCoefficientMap;
    double m_constant = 0;

public:
    LinearVariableComposition() = default;
    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    LinearVariableComposition(double constant);
    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    LinearVariableComposition(const Variable& variable);

    [[nodiscard]] auto const& GetVariableIdCoefficientMap() const
    {
        return m_variableIdCoefficientMap;
    }
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
