#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"

#include <CoinFinite.hpp>

tsplp::LinearConstraint tsplp::operator<=(LinearVariableComposition&& lhs, LinearVariableComposition&& rhs)
{
    LinearConstraint result = std::move(lhs) - std::move(rhs);
    result.m_upperBound *= -1;
    result.m_lowerBound = -COIN_DBL_MAX;
    return result;
}

tsplp::LinearConstraint tsplp::operator>=(LinearVariableComposition&& lhs, LinearVariableComposition&& rhs)
{
    LinearConstraint result = std::move(lhs) - std::move(rhs);
    result.m_lowerBound = -result.m_upperBound;
    result.m_upperBound = COIN_DBL_MAX;
    return result;
}

tsplp::LinearConstraint tsplp::operator==(LinearVariableComposition&& lhs, LinearVariableComposition&& rhs)
{
    LinearConstraint result = std::move(lhs) - std::move(rhs);
    result.m_upperBound *= -1;
    result.m_lowerBound = result.m_lowerBound;
    return result;
}

tsplp::LinearConstraint::LinearConstraint(LinearVariableComposition&& convertee)
    : m_coefficientMap(std::move(convertee.m_coefficientMap)), m_upperBound(convertee.m_constant)
{
}

bool tsplp::LinearConstraint::Evaluate(double tolerance) const
{
    double value = 0.0;
    for (auto const& [var, coef] : m_coefficientMap)
        value += coef * var.GetObjectiveValue();

    return m_lowerBound <= value + tolerance && value - tolerance <= m_upperBound;
}
