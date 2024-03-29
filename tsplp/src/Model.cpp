#include "Model.hpp"

#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"

#include <ClpSimplex.hpp>

#include <boost/range/iterator_range.hpp>

#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

tsplp::Model::Model() = default;

tsplp::Model::Model(size_t numberOfBinaryVariables)
    : m_spSimplexModel { std::make_unique<ClpSimplex>() }
    , m_spModelMutex { std::make_unique<std::mutex>() }
    , m_numberOfBinaryVariables(numberOfBinaryVariables)
{
    if (numberOfBinaryVariables > std::numeric_limits<int>::max())
        throw std::runtime_error("Too many variables");

    std::unique_lock lock { *m_spModelMutex };

    m_spSimplexModel->setLogLevel(0);
    m_spSimplexModel->addColumns(
        static_cast<int>(numberOfBinaryVariables), nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr);

    for (size_t i = 0; i < numberOfBinaryVariables; ++i)
        AddVariable(0.0, 1.0);
}

tsplp::Model::~Model() noexcept = default;

tsplp::Model::Model(const Model& other)
    : m_spSimplexModel(std::make_unique<ClpSimplex>(*other.m_spSimplexModel))
    , m_spModelMutex { std::make_unique<std::mutex>() }
    , m_variables(other.m_variables)
    , m_numberOfBinaryVariables(other.m_numberOfBinaryVariables)
{
}

tsplp::Model::Model(Model&& other) noexcept
    : m_spSimplexModel(std::move(other.m_spSimplexModel))
    , m_spModelMutex(std::move(other.m_spModelMutex))
    , m_variables(std::move(other.m_variables))
    , m_numberOfBinaryVariables(other.m_numberOfBinaryVariables)
{
}

tsplp::Model& tsplp::Model::operator=(Model other)
{
    swap(*this, other);
    return *this;
}

std::span<const tsplp::Variable> tsplp::Model::GetBinaryVariables() const
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return { m_variables.data(), m_variables.data() + m_numberOfBinaryVariables };
}

void tsplp::Model::SetObjective(const LinearVariableComposition& objective)
{
    std::unique_lock lock { *m_spModelMutex };

    m_spSimplexModel->setObjectiveOffset(-objective.GetConstant()); // offset is negative

    for (const auto& [varId, coef] : objective.GetVariableIdCoefficientMap())
        m_spSimplexModel->setObjectiveCoefficient(static_cast<int>(varId), coef);
}

template <typename RandIterator>
void tsplp::Model::AddConstraints(RandIterator first, RandIterator last)
{
    const auto numberOfConstraints = static_cast<size_t>(last - first);
    if (numberOfConstraints > std::numeric_limits<int>::max())
        throw std::runtime_error("Too many constraints");

    std::vector<double> lowerBounds;
    std::vector<double> upperBounds;
    lowerBounds.reserve(numberOfConstraints);
    upperBounds.reserve(numberOfConstraints);

    std::vector<int> rowStarts;
    rowStarts.reserve(numberOfConstraints + 1);
    rowStarts.push_back(0);

    std::vector<int> columns;
    std::vector<double> elements;
    for (const auto& c : boost::make_iterator_range(first, last))
    {
        lowerBounds.push_back(c.GetLowerBound());
        upperBounds.push_back(c.GetUpperBound());

        rowStarts.push_back(
            rowStarts.back() + static_cast<int>(std::ssize(c.GetVariableIdCoefficientMap())));

        for (const auto& [varId, coef] : c.GetVariableIdCoefficientMap())
        {
            elements.push_back(coef);
            columns.push_back(static_cast<int>(varId));
        }
    }

    std::unique_lock lock { *m_spModelMutex };

    m_spSimplexModel->addRows(
        static_cast<int>(numberOfConstraints), lowerBounds.data(), upperBounds.data(),
        rowStarts.data(), columns.data(), elements.data());
}

// do needed instantiations explicitly so the code can stay in cpp file
template void tsplp::Model::AddConstraints(
    std::deque<tsplp::LinearConstraint>::const_iterator first,
    std::deque<tsplp::LinearConstraint>::const_iterator last);
template void tsplp::Model::AddConstraints(
    std::vector<tsplp::LinearConstraint>::const_iterator first,
    std::vector<tsplp::LinearConstraint>::const_iterator last);

tsplp::Variable tsplp::Model::AddVariable(double lowerBound, double upperBound)
{
    m_variables.emplace_back(m_variables.size());
    if (m_variables.size() >= static_cast<size_t>(m_spSimplexModel->getNumCols()))
        m_spSimplexModel->addColumn(m_spSimplexModel->getNumCols(), nullptr, nullptr);

    m_variables.back().SetLowerBound(lowerBound, *this);
    m_variables.back().SetUpperBound(upperBound, *this);
    return m_variables.back();
}

tsplp::Status tsplp::Model::Solve(std::chrono::steady_clock::time_point endTime)
{
    const std::chrono::duration<double> remainingTime = endTime - std::chrono::steady_clock::now();

    std::unique_lock lock { *m_spModelMutex };

    m_spSimplexModel->setMaximumSeconds(remainingTime.count());
    m_spSimplexModel->dual();

    const auto modelStatus = m_spSimplexModel->status();

    switch (modelStatus)
    {
    case 0:
        return Status::Optimal;
    case 1:
        return Status::Infeasible;
    case 2:
        return Status::Unbounded;
    case 3:
        return Status::Timeout;
    default:
        return Status::Error;
    }
}

void tsplp::swap(tsplp::Model& m1, tsplp::Model& m2) noexcept
{
    using std::swap;

    swap(m1.m_spSimplexModel, m2.m_spSimplexModel);
    swap(m1.m_spModelMutex, m2.m_spModelMutex);
    swap(m1.m_variables, m2.m_variables);
    swap(m1.m_numberOfBinaryVariables, m2.m_numberOfBinaryVariables);
}
