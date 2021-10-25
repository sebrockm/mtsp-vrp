#include "LinearVariableComposition.hpp"
#include "Variable.hpp"

tsplp::LinearVariableComposition tsplp::operator*(double coef, Variable const& var)
{
    LinearVariableComposition result;
    result.m_coefficientMap[var] = coef;
    return result;
}

tsplp::LinearVariableComposition tsplp::operator*(double factor, LinearVariableComposition&& linearComp)
{
    for (auto&& [var, coef] : linearComp.m_coefficientMap)
        coef *= factor;

    return std::move(linearComp);
}

tsplp::LinearVariableComposition tsplp::operator+(LinearVariableComposition&& lhs, LinearVariableComposition&& rhs)
{
    auto&& biggerOne = std::move(lhs.m_coefficientMap.size() > rhs.m_coefficientMap.size() ? lhs : rhs);
    auto&& smallerOne = std::move(lhs.m_coefficientMap.size() > rhs.m_coefficientMap.size() ? rhs : lhs);

    for (auto&& [var, coef] : smallerOne.m_coefficientMap)
        biggerOne.m_coefficientMap[var] += coef;

    biggerOne.m_constant += smallerOne.m_constant;

    return std::move(biggerOne);
}

tsplp::LinearVariableComposition tsplp::operator+(LinearVariableComposition&& lhs, double rhs)
{
    lhs.m_constant += rhs;
    return std::move(lhs);
}

tsplp::LinearVariableComposition tsplp::operator-(LinearVariableComposition&& operand)
{
    for (auto&& [var, coef] : operand.m_coefficientMap)
        operand.m_coefficientMap[var] *= -1.0;

    operand.m_constant *= -1.0;

    return std::move(operand);
}

tsplp::LinearVariableComposition tsplp::operator-(LinearVariableComposition&& lhs, LinearVariableComposition&& rhs)
{
    return std::move(lhs) + (-std::move(rhs));
}

tsplp::LinearVariableComposition::LinearVariableComposition(double constant)
    : m_constant(constant)
{
}

tsplp::LinearVariableComposition::LinearVariableComposition(const Variable& variable)
    : m_coefficientMap{ {variable, 1} }
{
}
