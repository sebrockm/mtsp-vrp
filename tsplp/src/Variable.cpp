#include "Variable.hpp"

#include <ClpSimplex.hpp>

tsplp::Variable::Variable(ClpSimplex& model, size_t id)
    : m_pModel(&model), m_id(id)
{
}

double tsplp::Variable::GetUpperBound() const
{
    return m_pModel->getColUpper()[m_id];
}

double tsplp::Variable::GetLowerBound() const
{
    return m_pModel->getColLower()[m_id];
}

void tsplp::Variable::SetUpperBound(double upperBound)
{
    m_pModel->setColumnUpper(static_cast<int>(m_id), upperBound);
}

void tsplp::Variable::SetLowerBound(double lowerBound)
{
    m_pModel->setColumnLower(static_cast<int>(m_id), lowerBound);
}

double tsplp::Variable::GetObjectiveValue() const
{
    return m_pModel->primalColumnSolution()[m_id];
}

size_t tsplp::Variable::GetId() const
{
    return m_id;
}

bool tsplp::VariableLess::operator()(const Variable& lhs, const Variable& rhs) const
{
    return lhs.GetId() < rhs.GetId();
}
