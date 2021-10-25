#pragma once

#include "Variable.hpp"

#include <map>

namespace tsplp
{
    class LinearVariableComposition
    {
        friend LinearVariableComposition operator*(double coef, const Variable& var);
        friend LinearVariableComposition operator*(double factor, LinearVariableComposition&& linearComp);

        friend LinearVariableComposition operator+(LinearVariableComposition&& lhs, LinearVariableComposition&& rhs);
        friend LinearVariableComposition operator+(LinearVariableComposition&& lhs, double rhs);

        friend LinearVariableComposition operator-(LinearVariableComposition&& operand);
        friend LinearVariableComposition operator-(LinearVariableComposition&& lhs, LinearVariableComposition&& rhs);

        friend class LinearConstraint;

    private:
        std::map<Variable, double, VariableLess> m_coefficientMap;
        double m_constant = 0;

    public:
        LinearVariableComposition() = default;
        LinearVariableComposition(double constant);
        LinearVariableComposition(const Variable& variable);
    };
}