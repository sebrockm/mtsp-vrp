#include "Model.hpp"

#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"

#include <ClpSimplex.hpp>

#include <boost/range/iterator_range.hpp>

#include <deque>
#include <future>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

tsplp::Model::Model(size_t numberOfBinaryVariables)
    : m_spSimplexModel { std::make_shared<ClpSimplex>() }
    , m_spModelMutex { std::make_shared<std::mutex>() }
    , m_variables {}
{
    if (numberOfBinaryVariables > std::numeric_limits<int>::max())
        throw std::runtime_error("Too many variables");

    std::unique_lock lock { *m_spModelMutex };

    m_spSimplexModel->setLogLevel(0);
    m_spSimplexModel->addColumns(
        static_cast<int>(numberOfBinaryVariables), nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr);

    m_variables.reserve(numberOfBinaryVariables);
    for (size_t i = 0; i < numberOfBinaryVariables; ++i)
    {
        m_variables.emplace_back(i);
        m_variables.back().SetLowerBound(0.0, *this);
        m_variables.back().SetUpperBound(1.0, *this);
    }
}

tsplp::Model::~Model() noexcept = default;

tsplp::Model::Model(const Model& other)
    : m_spSimplexModel(std::make_shared<ClpSimplex>(*other.m_spSimplexModel))
    , m_spModelMutex { std::make_shared<std::mutex>() }
    , m_variables(other.m_variables)
{
}

tsplp::Model::Model(Model&& other) noexcept
    : m_spSimplexModel(std::move(other.m_spSimplexModel))
    , m_spModelMutex(std::move(other.m_spModelMutex))
    , m_variables(std::move(other.m_variables))
{
}

tsplp::Model& tsplp::Model::operator=(Model other)
{
    swap(*this, other);
    return *this;
}

void tsplp::Model::SetObjective(const LinearVariableComposition& objective)
{
    std::unique_lock lock { *m_spModelMutex };

    m_spSimplexModel->setObjectiveOffset(-objective.GetConstant()); // offset is negative

    for (size_t i = 0; i < objective.GetCoefficients().size(); ++i)
        m_spSimplexModel->setObjectiveCoefficient(
            static_cast<int>(objective.GetVariables()[i].GetId()), objective.GetCoefficients()[i]);
}

template <typename RandIterator>
void tsplp::Model::AddConstraints(RandIterator first, RandIterator last)
{
    const auto numberOfConstraints = static_cast<size_t>(last - first);
    if (numberOfConstraints > std::numeric_limits<int>::max())
        throw std::runtime_error("Too many constraints");

    std::vector<double> lowerBounds, upperBounds;
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

        rowStarts.push_back(rowStarts.back() + static_cast<int>(std::ssize(c.GetCoefficients())));

        for (const auto coef : c.GetCoefficients())
            elements.push_back(coef);
        for (const auto var : c.GetVariables())
            columns.push_back(static_cast<int>(var.GetId()));
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

tsplp::Status tsplp::Model::Solve(std::chrono::steady_clock::time_point endTime)
{
    const auto solverTask = [](std::shared_ptr<ClpSimplex> spSimplexModel,
                               std::shared_ptr<std::mutex> spModelMutex, std::promise<void> promise)
    {
        std::unique_lock lock { *spModelMutex };

        spSimplexModel->dual();
        promise.set_value();
    };

    std::promise<void> solverPromise;
    const auto solverFuture = solverPromise.get_future();

    std::thread solverThread(
        solverTask, m_spSimplexModel, m_spModelMutex, std::move(solverPromise));

    const auto futureStatus = solverFuture.wait_until(endTime);

    if (futureStatus == std::future_status::timeout)
    {
        solverThread.detach();
        return Status::Timeout;
    }

    solverThread.join();

    const auto modelStatus = m_spSimplexModel->status();

    switch (modelStatus)
    {
    case 0:
        return Status::Optimal;
    case 1:
        return Status::Infeasible;
    case 2:
        return Status::Unbounded;
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
}
