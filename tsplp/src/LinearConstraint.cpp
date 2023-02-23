#include "LinearConstraint.hpp"

#include "LinearVariableComposition.hpp"

#include <limits>

tsplp::LinearConstraint tsplp::operator<=(
    LinearVariableComposition lhs, LinearVariableComposition rhs)
{
    LinearConstraint result { std::move(lhs) - std::move(rhs) };
    result.m_upperBound *= -1;
    result.m_lowerBound = -std::numeric_limits<double>::max();
    return result;
}

tsplp::LinearConstraint tsplp::operator>=(
    LinearVariableComposition lhs, LinearVariableComposition rhs)
{
    LinearConstraint result { std::move(lhs) - std::move(rhs) };
    result.m_lowerBound = -result.m_upperBound;
    result.m_upperBound = std::numeric_limits<double>::max();
    return result;
}

tsplp::LinearConstraint tsplp::operator==(
    LinearVariableComposition lhs, LinearVariableComposition rhs)
{
    LinearConstraint result { std::move(lhs) - std::move(rhs) };
    result.m_upperBound *= -1;
    result.m_lowerBound = result.m_upperBound;
    return result;
}

tsplp::LinearConstraint::LinearConstraint(LinearVariableComposition&& convertee)
    : m_variables(std::move(convertee.m_variables))
    , m_coefficients(std::move(convertee.m_coefficients))
    , m_upperBound(convertee.m_constant)
{
}

bool tsplp::LinearConstraint::Evaluate(const Model& model, double tolerance) const
{
    double value = 0.0;
    for (size_t i = 0; i < m_coefficients.size(); ++i)
        value += m_coefficients[i] * m_variables[i].GetObjectiveValue(model);

    return m_lowerBound <= value + tolerance && value - tolerance <= m_upperBound;
}
