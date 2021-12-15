#pragma once

#include "Variable.hpp"

#include <vector>

namespace tsplp
{
    class LinearVariableComposition
    {
        friend LinearVariableComposition operator*(double factor, LinearVariableComposition linearComp);

        friend LinearVariableComposition operator+(LinearVariableComposition lhs, LinearVariableComposition rhs);
        friend LinearVariableComposition operator+(LinearVariableComposition lhs, double rhs);

        friend LinearVariableComposition& operator+=(LinearVariableComposition& lhs, LinearVariableComposition const& rhs);

        friend LinearVariableComposition operator-(LinearVariableComposition operand);
        friend LinearVariableComposition operator-(LinearVariableComposition lhs, LinearVariableComposition rhs);

        friend class LinearConstraint;

    private:
        std::vector<Variable> m_variables;
        std::vector<double> m_coefficients;
        double m_constant = 0;

    public:
        LinearVariableComposition() = default;
        LinearVariableComposition(double constant);
        LinearVariableComposition(const Variable& variable);

        auto const& GetVariables() const { return m_variables; }
        auto const& GetCoefficients() const { return m_coefficients; }
        double GetConstant() const { return m_constant; }

        double Evaluate() const;
    };

    LinearVariableComposition operator*(double factor, LinearVariableComposition linearComp);

    LinearVariableComposition operator+(LinearVariableComposition lhs, LinearVariableComposition rhs);
    LinearVariableComposition operator+(LinearVariableComposition lhs, double rhs);

    LinearVariableComposition& operator+=(LinearVariableComposition& lhs, LinearVariableComposition const& rhs);

    LinearVariableComposition operator-(LinearVariableComposition operand);
    LinearVariableComposition operator-(LinearVariableComposition lhs, LinearVariableComposition rhs);
}