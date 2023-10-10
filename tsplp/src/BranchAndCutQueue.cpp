#include "BranchAndCutQueue.hpp"

#include "Model.hpp"

#include <algorithm>
#include <iostream>

tsplp::BranchAndCutQueue::BranchAndCutQueue(size_t threadCount)
    : m_workedOnLowerBounds(threadCount)
{
    if (threadCount == 0)
        throw std::logic_error("Cannot have zero threads");
}

double tsplp::BranchAndCutQueue::GetLowerBound() const
{
    std::unique_lock lock { m_mutex };
    return CalculateLowerBound();
}

std::optional<std::tuple<tsplp::SData, tsplp::NodeDoneNotifier>> tsplp::BranchAndCutQueue::Pop(
    size_t threadId)
{
    if (threadId >= m_workedOnLowerBounds.size())
        throw std::logic_error("Wrong threadId");

    std::unique_lock lock { m_mutex };

    while (!m_isCleared && m_heap.empty() && m_workedOnCount > 0)
        m_cv.wait(lock);

    if (m_isCleared || (m_heap.empty() && m_workedOnCount == 0))
        return std::nullopt;

    std::pop_heap(begin(m_heap), end(m_heap), m_comparer);
    auto popped = std::move(m_heap.back());
    m_heap.pop_back();

    m_workedOnLowerBounds.at(threadId) = popped.LowerBound;
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

    if (!m_workedOnLowerBounds.at(threadId).has_value())
        throw std::logic_error("Thread does not have a node popped");

    if (currentLowerBound < *m_workedOnLowerBounds.at(threadId))
        throw std::logic_error("Lower bound must not be decreased");

    m_workedOnLowerBounds.at(threadId) = currentLowerBound;
}

void tsplp::BranchAndCutQueue::Push(
    double lowerBound, std::vector<Variable> fixedVariables0, std::vector<Variable> fixedVariables1)
{
    bool needsNotify = false;

    {
        std::unique_lock lock { m_mutex };

        if (m_isCleared)
            return;

        if (lowerBound < CalculateLowerBound())
            throw std::logic_error("cannot push smaller lower bound");

        needsNotify = m_heap.empty() && m_workedOnCount > 0;

        m_heap.push_back({ lowerBound, std::move(fixedVariables0), std::move(fixedVariables1) });
        std::push_heap(begin(m_heap), end(m_heap), m_comparer);
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

        if (m_isCleared)
            return;

        if (lowerBound < CalculateLowerBound())
            throw std::logic_error("cannot push smaller lower bound");

        needsNotify = m_heap.empty() && m_workedOnCount > 0;

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
        if (!m_workedOnLowerBounds.at(threadId).has_value())
            throw std::logic_error("Thread does not have a node popped");

        m_workedOnLowerBounds.at(threadId) = std::nullopt;
        --m_workedOnCount;
        needsNotify = m_workedOnCount == 0 && !m_isCleared && m_heap.empty();
    }

    if (needsNotify)
        m_cv.notify_all();
}

double tsplp::BranchAndCutQueue::CalculateLowerBound() const
{
    static constexpr auto max = std::numeric_limits<double>::max();

    if (m_heap.empty() && m_workedOnCount == 0)
        return -max;

    const auto cmp = [](auto opt1, auto opt2) { return opt1.value_or(max) < opt2.value_or(max); };

    const auto lbHeap = m_heap.empty() ? max : m_heap.front().LowerBound;

    const auto lbWorkedOn = m_workedOnCount == 0
        ? max
        : std::min_element(m_workedOnLowerBounds.begin(), m_workedOnLowerBounds.end(), cmp)
              ->value();

    return std::min(lbHeap, lbWorkedOn);
}

void tsplp::BranchAndCutQueue::Print() const
{
    std::unique_lock lock { m_mutex };

    std::cout << "BranchAndCutQueue:\nHeap:";
    for (const auto& [lb, v0, v1] : m_heap)
    {
        std::cout << "LB:" << lb << ", v0:";
        for (const auto& v : v0)
            std::cout << v.GetId() << ",";
        std::cout << " v1:";
        for (const auto& v : v1)
            std::cout << v.GetId() << ",";
        std::cout << std::endl;
    }

    std::cout << "\nworked on LBs: ";
    for (size_t i = 0; i < m_workedOnLowerBounds.size(); ++i)
    {
        if (m_workedOnLowerBounds[i])
        {
            std::cout << i << ":" << *m_workedOnLowerBounds[i];
        }
        else
        {
            std::cout << i << ":<>";
        }
        std::cout << " ";
    }

    std::cout << "\nis cleared: " << m_isCleared << std::endl << std::endl;
}
