#include "Variable.hpp"

#include "Model.hpp"

#include <ClpSimplex.hpp>

tsplp::Variable::Variable(size_t id)
    : m_id(id)
{
}

double tsplp::Variable::GetUpperBound(const Model& model) const
{
    return model.m_spSimplexModel->getColUpper()[m_id];
}

double tsplp::Variable::GetLowerBound(const Model& model) const
{
    return model.m_spSimplexModel->getColLower()[m_id];
}

void tsplp::Variable::SetUpperBound(double upperBound, Model& model) const
{
    model.m_spSimplexModel->setColumnUpper(static_cast<int>(m_id), upperBound);
}

void tsplp::Variable::SetLowerBound(double lowerBound, Model& model) const
{
    model.m_spSimplexModel->setColumnLower(static_cast<int>(m_id), lowerBound);
}

double tsplp::Variable::GetObjectiveValue(const Model& model) const
{
    return model.m_spSimplexModel->primalColumnSolution()[m_id];
}

size_t tsplp::Variable::GetId() const { return m_id; }

bool tsplp::VariableLess::operator()(const Variable& lhs, const Variable& rhs) const
{
    return lhs.GetId() < rhs.GetId();
}
