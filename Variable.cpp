#include "Variable.hpp"

#include <ClpSimplex.hpp>

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
