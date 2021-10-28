#pragma once

#include "Variable.hpp"

#include <map>

namespace tsplp
{
    class LinearVariableComposition;

    class LinearConstraint
    {
        friend LinearConstraint operator<=(LinearVariableComposition&& lhs, LinearVariableComposition&& rhs);
        friend LinearConstraint operator>=(LinearVariableComposition&& lhs, LinearVariableComposition&& rhs);
        friend LinearConstraint operator==(LinearVariableComposition&& lhs, LinearVariableComposition&& rhs);

    private:
        std::map<Variable, double, VariableLess> m_coefficientMap;
        double m_upperBound = 0.0;
        double m_lowerBound = 0.0;

    private:
        LinearConstraint(LinearVariableComposition&& convertee);

    public:
        double GetUpperBound() const { return m_upperBound; }
        double GetLowerBound() const { return m_lowerBound; }
        auto const& GetCoefficientMap() const { return m_coefficientMap; }
    };
}