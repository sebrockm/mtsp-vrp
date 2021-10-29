#include "Variable.hpp"

#include <ClpSimplex.hpp>

tsplp::Variable::Variable(ClpSimplex& model, int id)
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
    m_pModel->setColumnUpper(m_id, upperBound);
}

void tsplp::Variable::SetLowerBound(double lowerBound)
{
    m_pModel->setColumnLower(m_id, lowerBound);
}

int tsplp::Variable::GetId() const
{
    return m_id;
}

bool tsplp::VariableLess::operator()(const Variable& lhs, const Variable& rhs) const
{
    return lhs.GetId() < rhs.GetId();
}

tsplp::Variables::Variables(ClpSimplex& model)
    : m_pModel(&model)
{
}

size_t tsplp::Variables::GetSize() const
{
    return static_cast<size_t>(m_pModel->getNumCols());
}

tsplp::Variable tsplp::Variables::operator[](int id) const
{
    if (id < 0 || id >= GetSize())
        throw std::runtime_error("Invalid id");

    return Variable(*m_pModel, id);
}

std::span<const double> tsplp::Variables::GetObjectiveValues() const
{
    return { m_pModel->primalColumnSolution(), GetSize() };
}
