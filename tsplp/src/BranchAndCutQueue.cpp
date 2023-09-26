#include "BranchAndCutQueue.hpp"

#include "Model.hpp"

#include <algorithm>
#include <cassert>
#include <map>

tsplp::BranchAndCutQueue::BranchAndCutQueue(size_t threadCount)
    : m_currentlyWorkedOnLowerBounds(threadCount)
{
    if (threadCount == 0)
        throw std::logic_error("Cannot have zero threads");
}

double tsplp::BranchAndCutQueue::GetLowerBound() const
{
    std::unique_lock lock { m_mutex };
    return m_lastFinishedLowerBound;
}

std::optional<std::tuple<tsplp::SData, tsplp::NodeDoneNotifier>> tsplp::BranchAndCutQueue::Pop(
    size_t threadId)
{
    if (threadId >= m_currentlyWorkedOnLowerBounds.size())
        throw std::logic_error("Wrong threadId");

    std::unique_lock lock { m_mutex };

    while (!m_isCleared && m_heap.empty() && m_workedOnCount > 0)
        m_cv.wait(lock);

    if (m_isCleared || (m_heap.empty() && m_workedOnCount == 0))
        return std::nullopt;

    std::pop_heap(begin(m_heap), end(m_heap), m_comparer);
    auto popped = std::move(m_heap.back());
    m_heap.pop_back();

    m_currentlyWorkedOnLowerBounds.at(threadId) = popped.LowerBound;
    ++m_finishedLowerBounds[popped.LowerBound];
    ++m_workedOnCount;

    return std::make_optional(std::make_tuple(
        std::move(popped), NodeDoneNotifier { [this, threadId] { NotifyNodeDone(threadId); } }));
}

void tsplp::BranchAndCutQueue::ClearAll()
{
    {
        std::unique_lock lock { m_mutex };
        m_isCleared = true;
    }

    m_cv.notify_all();
}

void tsplp::BranchAndCutQueue::UpdateCurrentLowerBound(size_t threadId, double currentLowerBound)
{
    std::unique_lock lock { m_mutex };

    if (!m_currentlyWorkedOnLowerBounds.at(threadId).has_value())
        throw std::logic_error("Thread does not have a node popped");

    if (currentLowerBound < *m_currentlyWorkedOnLowerBounds.at(threadId))
        throw std::logic_error("Lower bound must not be decreased");

    if (currentLowerBound > *m_currentlyWorkedOnLowerBounds.at(threadId))
    {
        DecreaseWorkedOn(*m_currentlyWorkedOnLowerBounds.at(threadId));
        m_currentlyWorkedOnLowerBounds.at(threadId) = currentLowerBound;
        ++m_finishedLowerBounds[currentLowerBound];
        m_lastFinishedLowerBound = CalculateLowerBound();
    }
}

void tsplp::BranchAndCutQueue::Push(
    double lowerBound, std::vector<Variable> fixedVariables0, std::vector<Variable> fixedVariables1)
{
    bool needsNotify = false;

    {
        std::unique_lock lock { m_mutex };

        if (lowerBound < m_lastFinishedLowerBound)
            throw std::logic_error("cannot push smaller lower bound");

        if (m_isCleared)
            return;

        needsNotify = m_heap.empty();

        m_heap.push_back({ lowerBound, std::move(fixedVariables0), std::move(fixedVariables1) });
        std::push_heap(begin(m_heap), end(m_heap), m_comparer);

        m_lastFinishedLowerBound = CalculateLowerBound();
    }

    if (needsNotify)
        m_cv.notify_one();
}

void tsplp::BranchAndCutQueue::PushBranch(
    double lowerBound, std::vector<Variable> fixedVariables0, std::vector<Variable> fixedVariables1,
    Variable branchingVariable, std::vector<Variable> recursivelyFixed0)
{
    bool needsNotify = false;

    {
        std::unique_lock lock { m_mutex };

        if (lowerBound < m_lastFinishedLowerBound)
            throw std::logic_error("cannot push smaller lower bound");

        if (m_isCleared)
            return;

        needsNotify = m_heap.empty();

        auto copyFixedVariables0 = fixedVariables0;
        auto copyFixedVariables1 = fixedVariables1;

        fixedVariables0.push_back(branchingVariable);
        copyFixedVariables1.push_back(branchingVariable);

        copyFixedVariables0.insert(
            copyFixedVariables0.end(), recursivelyFixed0.begin(), recursivelyFixed0.end());

        m_heap.push_back({ lowerBound, std::move(fixedVariables0), std::move(fixedVariables1) });
        std::push_heap(begin(m_heap), end(m_heap), m_comparer);

        m_heap.push_back(
            { lowerBound, std::move(copyFixedVariables0), std::move(copyFixedVariables1) });
        std::push_heap(begin(m_heap), end(m_heap), m_comparer);

        m_lastFinishedLowerBound = CalculateLowerBound();
    }

    if (needsNotify)
    {
        m_cv.notify_one();
        m_cv.notify_one();
    }
}

void tsplp::BranchAndCutQueue::NotifyNodeDone(size_t threadId)
{
    bool needsNotify = false;

    {
        std::unique_lock lock { m_mutex };
        if (!m_currentlyWorkedOnLowerBounds.at(threadId).has_value())
            throw std::logic_error("Thread does not have a node popped");

        DecreaseWorkedOn(*m_currentlyWorkedOnLowerBounds.at(threadId));
        m_currentlyWorkedOnLowerBounds.at(threadId) = std::nullopt;
        --m_workedOnCount;
        needsNotify = m_workedOnCount == 0;

        m_lastFinishedLowerBound = CalculateLowerBound();
    }

    if (needsNotify)
        m_cv.notify_all();
}

double tsplp::BranchAndCutQueue::CalculateLowerBound() const
{
    const auto lbHeap
        = m_heap.empty() ? std::numeric_limits<double>::infinity() : m_heap.front().LowerBound;
    const auto lbUsed = m_finishedLowerBounds.empty() ? std::numeric_limits<double>::infinity()
                                                      : m_finishedLowerBounds.begin()->first;

    return std::min(lbHeap, lbUsed);
}

void tsplp::BranchAndCutQueue::DecreaseWorkedOn(double lb)
{
    assert(m_finishedLowerBounds.at(lb) > 0);
    --m_finishedLowerBounds.at(lb);

    while (!m_finishedLowerBounds.empty() && m_finishedLowerBounds.begin()->second == 0)
    {
        m_lastFinishedLowerBound = m_finishedLowerBounds.begin()->first;
        m_finishedLowerBounds.erase(m_finishedLowerBounds.begin());
    }
}
