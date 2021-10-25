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
