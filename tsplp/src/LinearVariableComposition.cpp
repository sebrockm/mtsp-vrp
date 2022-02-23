#include "LinearVariableComposition.hpp"
#include "Variable.hpp"

#include <algorithm>

tsplp::LinearVariableComposition tsplp::operator*(double factor, LinearVariableComposition linearComp)
{
    for (auto& coef : linearComp.m_coefficients)
        coef *= factor;

    linearComp.m_constant *= factor;

    return linearComp;
}

tsplp::LinearVariableComposition tsplp::operator+(LinearVariableComposition lhs, LinearVariableComposition rhs)
{
    auto& biggerOne = lhs.m_coefficients.size() > rhs.m_coefficients.size() ? lhs : rhs;
    const auto& smallerOne = lhs.m_coefficients.size() > rhs.m_coefficients.size() ? rhs : lhs;

    return std::move(biggerOne += smallerOne);
}

tsplp::LinearVariableComposition tsplp::operator+(LinearVariableComposition lhs, double rhs)
{
    lhs.m_constant += rhs;
    return lhs;
}

tsplp::LinearVariableComposition& tsplp::operator+=(LinearVariableComposition& lhs, LinearVariableComposition const& rhs)
{
    for (size_t i = 0; i < rhs.m_variables.size(); ++i)
    {
        const auto iter = std::upper_bound(lhs.m_variables.begin(), lhs.m_variables.end(), rhs.m_variables[i], VariableLess{});

        if (iter != lhs.m_variables.end() && iter->GetId() == rhs.m_variables[i].GetId())
        {
            const auto id = static_cast<size_t>(iter - lhs.m_variables.begin());
            lhs.m_coefficients[id] += rhs.m_coefficients[i];
        }
        else
        {
            const auto coefIter = lhs.m_coefficients.begin() + (iter - lhs.m_variables.begin());
            lhs.m_variables.insert(iter, rhs.m_variables[i]);
            lhs.m_coefficients.insert(coefIter, rhs.m_coefficients[i]);
        }
    }

    lhs.m_constant += rhs.m_constant;

    return lhs;
}

tsplp::LinearVariableComposition tsplp::operator-(LinearVariableComposition operand)
{
    for (auto& coef : operand.m_coefficients)
        coef *= -1.0;

    operand.m_constant *= -1.0;

    return operand;
}

tsplp::LinearVariableComposition tsplp::operator-(LinearVariableComposition lhs, LinearVariableComposition rhs)
{
    return std::move(lhs) + (-std::move(rhs));
}

tsplp::LinearVariableComposition::LinearVariableComposition(double constant)
    : m_constant(constant)
{
}

tsplp::LinearVariableComposition::LinearVariableComposition(const Variable& variable)
    : m_variables{ variable }, m_coefficients{ 1 }
{
}

double tsplp::LinearVariableComposition::Evaluate() const
{
    double result = m_constant;
    for (size_t i = 0; i < m_coefficients.size(); ++i)
        result += m_coefficients[i] * m_variables[i].GetObjectiveValue();

    return result;
}
