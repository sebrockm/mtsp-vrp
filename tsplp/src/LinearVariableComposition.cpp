#include "LinearVariableComposition.hpp"

#include "Variable.hpp"

#include <algorithm>

tsplp::LinearVariableComposition tsplp::operator*(
    double factor, LinearVariableComposition linearComp)
{
    for (auto& [varId, coef] : linearComp.m_variableIdCoefficientMap)
        coef *= factor;

    linearComp.m_constant *= factor;

    return linearComp;
}

tsplp::LinearVariableComposition tsplp::operator+(
    LinearVariableComposition lhs, LinearVariableComposition rhs)
{
    auto& biggerOne
        = lhs.m_variableIdCoefficientMap.size() > rhs.m_variableIdCoefficientMap.size() ? lhs : rhs;
    const auto& smallerOne
        = lhs.m_variableIdCoefficientMap.size() > rhs.m_variableIdCoefficientMap.size() ? rhs : lhs;

    return std::move(biggerOne += smallerOne);
}

tsplp::LinearVariableComposition tsplp::operator+(LinearVariableComposition lhs, double rhs)
{
    lhs.m_constant += rhs;
    return lhs;
}

tsplp::LinearVariableComposition& tsplp::operator+=(
    LinearVariableComposition& lhs, LinearVariableComposition const& rhs)
{
    for (const auto& [varId, coef] : rhs.m_variableIdCoefficientMap)
    {
        if (const auto iter = lhs.m_variableIdCoefficientMap.find(varId);
            iter != lhs.m_variableIdCoefficientMap.end())
        {
            iter->second += coef;
        }
        else
        {
            lhs.m_variableIdCoefficientMap.emplace(varId, coef);
        }
    }

    lhs.m_constant += rhs.m_constant;

    return lhs;
}

tsplp::LinearVariableComposition tsplp::operator-(LinearVariableComposition operand)
{
    for (auto& [varId, coef] : operand.m_variableIdCoefficientMap)
        coef *= -1.0;

    operand.m_constant *= -1.0;

    return operand;
}

tsplp::LinearVariableComposition tsplp::operator-(
    LinearVariableComposition lhs, LinearVariableComposition rhs)
{
    return std::move(lhs) + (-std::move(rhs));
}

tsplp::LinearVariableComposition::LinearVariableComposition(double constant)
    : m_constant(constant)
{
}

tsplp::LinearVariableComposition::LinearVariableComposition(const Variable& variable)
    : m_variableIdCoefficientMap { { variable.GetId(), 1.0 } }
{
}

double tsplp::LinearVariableComposition::Evaluate(const Model& model) const
{
    double result = m_constant;
    for (const auto& [varId, coef] : m_variableIdCoefficientMap)
        result += coef * Variable { varId }.GetObjectiveValue(model);

    return result;
}
