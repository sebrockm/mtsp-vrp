#include "Model.hpp"
#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"

#include <ClpSimplex.hpp>

#include <vector>

tsplp::Model::Model(size_t numberOfBinaryVariables)
    : m_spSimplexModel{std::make_unique<ClpSimplex>()}
{
    std::vector<double> lowerBounds(0.0, numberOfBinaryVariables);
    std::vector<double> upperBounds(1.0, numberOfBinaryVariables);

    m_spSimplexModel->addColumns(numberOfBinaryVariables, lowerBounds.data(), upperBounds.data(), nullptr, nullptr, nullptr, nullptr);
}

tsplp::Variables tsplp::Model::GetVariables() const
{
    return Variables(*m_spSimplexModel);
}

void tsplp::Model::SetObjective(const LinearVariableComposition& objective)
{
    m_spSimplexModel->setObjectiveOffset(-objective.GetConstant()); // offset is negative

    for (auto const& [var, coef] : objective.GetCoefficientMap())
        m_spSimplexModel->setObjectiveCoefficient(var.GetId(), coef);
}

void tsplp::Model::AddConstraints(std::span<const LinearConstraint> constraints)
{
    std::vector<double> lowerBounds, upperBounds;
    lowerBounds.reserve(constraints.size());
    upperBounds.reserve(constraints.size());

    std::vector<int> rowStarts;
    rowStarts.reserve(constraints.size() + 1);
    rowStarts.push_back(0);

    std::vector<int> columns;
    std::vector<double> elements;

    for (const auto& c : constraints)
    {
        lowerBounds.push_back(c.GetLowerBound());
        upperBounds.push_back(c.GetUpperBound());

        rowStarts.push_back(rowStarts.back() + c.GetCoefficientMap().size());
        for (auto const& [var, coef] : c.GetCoefficientMap())
        {
            columns.push_back(var.GetId());
            elements.push_back(coef);
        }
    }

    m_spSimplexModel->addRows(std::ssize(constraints), lowerBounds.data(), upperBounds.data(), rowStarts.data(), columns.data(), elements.data());
}

tsplp::Status tsplp::Model::Solve()
{
    m_spSimplexModel->primal();
    const auto status = m_spSimplexModel->status();

    switch (status)
    {
    case 0: return Status::Optimal;
    case 1: return Status::Infeasible;
    case 2: return Status::Unbounded;
    default: return Status::Error;
    }
}