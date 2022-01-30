#include "Model.hpp"
#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"

#include <ClpSimplex.hpp>
#include <limits>
#include <stdexcept>

tsplp::Model::Model(size_t numberOfBinaryVariables)
    : m_spSimplexModel{ std::make_unique<ClpSimplex>() }, m_variables{}
{
    if (numberOfBinaryVariables > std::numeric_limits<int>::max())
        throw std::runtime_error("Too many variables");

    m_spSimplexModel->setLogLevel(0);
    m_spSimplexModel->addColumns(static_cast<int>(numberOfBinaryVariables), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    m_variables.reserve(numberOfBinaryVariables);
    for (size_t i = 0; i < numberOfBinaryVariables; ++i)
    {
        m_variables.emplace_back(*m_spSimplexModel, i);
        m_variables.back().SetLowerBound(0.0);
        m_variables.back().SetUpperBound(1.0);
    }
}

tsplp::Model::~Model()
{
    // ClpSimplex gets destroyed here so that it can stay forward declared in header file
}

tsplp::Model::Model(const Model& other)
    : m_spSimplexModel(std::make_unique<ClpSimplex>(*other.m_spSimplexModel))
{
    m_variables.reserve(other.m_variables.size());

    for (auto v : other.m_variables)
        m_variables.emplace_back(*m_spSimplexModel, v.GetId());
}

tsplp::Model::Model(Model&& other) noexcept
    : m_spSimplexModel(std::move(other.m_spSimplexModel)), m_variables(std::move(other.m_variables))
{
}

tsplp::Model& tsplp::Model::operator=(Model other)
{
    swap(*this, other);
    return *this;
}

void tsplp::Model::SetObjective(const LinearVariableComposition& objective)
{
    m_spSimplexModel->setObjectiveOffset(-objective.GetConstant()); // offset is negative

    for (size_t i = 0; i < objective.GetCoefficients().size(); ++i)
        m_spSimplexModel->setObjectiveCoefficient(static_cast<int>(objective.GetVariables()[i].GetId()), objective.GetCoefficients()[i]);
}

void tsplp::Model::AddConstraints(std::span<const LinearConstraint> constraints)
{
    if (std::ssize(constraints) > std::numeric_limits<int>::max())
        throw std::runtime_error("Too many constraints");

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

        rowStarts.push_back(rowStarts.back() + static_cast<int>(std::ssize(c.GetCoefficients())));

        for (const auto coef : c.GetCoefficients())
            elements.push_back(coef);
        for (const auto var : c.GetVariables())
            columns.push_back(static_cast<int>(var.GetId()));
    }

    m_spSimplexModel->addRows(static_cast<int>(std::ssize(constraints)), lowerBounds.data(), upperBounds.data(), rowStarts.data(), columns.data(), elements.data());
}

tsplp::Status tsplp::Model::Solve()
{
    m_spSimplexModel->dual();
    const auto status = m_spSimplexModel->status();

    switch (status)
    {
    case 0: return Status::Optimal;
    case 1: return Status::Infeasible;
    case 2: return Status::Unbounded;
    default: return Status::Error;
    }
}

void tsplp::swap(tsplp::Model& m1, tsplp::Model& m2) noexcept
{
    using std::swap;

    swap(m1.m_spSimplexModel, m2.m_spSimplexModel);
    swap(m1.m_variables, m2.m_variables);
}