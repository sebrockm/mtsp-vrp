#include "BranchAndCutQueue.hpp"

#include "Model.hpp"

#include <algorithm>
#include <cassert>

tsplp::BranchAndCutQueue::BranchAndCutQueue() { m_heap.emplace_back(); }

void tsplp::BranchAndCutQueue::ClearAll()
{
    {
        std::unique_lock lock { m_mutex };
        m_isCleared = true;
    }

    m_cv.notify_all();
}

void tsplp::BranchAndCutQueue::NotifyNodeDone(size_t threadId)
{
    bool needsNotify = false;

    {
        std::unique_lock lock { m_mutex };
        [[maybe_unused]] const auto removedCount = m_currentlyWorkedOnLowerBounds.erase(threadId);
        assert(removedCount == 1);
        needsNotify = m_currentlyWorkedOnLowerBounds.empty();
    }

    if (needsNotify)
        m_cv.notify_all();
}

std::optional<double> tsplp::BranchAndCutQueue::GetLowerBound() const
{
    std::unique_lock lock { m_mutex };

    const auto lbHeap
        = m_heap.empty() ? std::nullopt : std::make_optional(m_heap.front().LowerBound);

    const auto minIter = std::min_element(
        begin(m_currentlyWorkedOnLowerBounds), end(m_currentlyWorkedOnLowerBounds),
        [](auto p1, auto p2) { return p1.second < p2.second; });
    const auto lbUsed = minIter == m_currentlyWorkedOnLowerBounds.end()
        ? std::nullopt
        : std::make_optional(minIter->second);

    if (lbHeap.has_value() && lbUsed.has_value())
        return std::min(lbHeap, lbUsed);

    if (lbHeap.has_value() && !lbUsed.has_value())
        return lbHeap;

    if (!lbHeap.has_value() && lbUsed.has_value())
        return lbUsed;

    return std::nullopt;
}

void tsplp::BranchAndCutQueue::UpdateCurrentLowerBound(size_t threadId, double currentLowerBound)
{
    std::unique_lock lock { m_mutex };

    assert(m_currentlyWorkedOnLowerBounds.contains(threadId));
    m_currentlyWorkedOnLowerBounds[threadId] = currentLowerBound;
}

size_t tsplp::BranchAndCutQueue::GetSize() const
{
    std::unique_lock lock { m_mutex };

    return m_heap.size();
}

size_t tsplp::BranchAndCutQueue::GetWorkedOnSize() const
{
    std::unique_lock lock { m_mutex };

    return m_currentlyWorkedOnLowerBounds.size();
}

std::optional<tsplp::SData> tsplp::BranchAndCutQueue::Pop(size_t threadId)
{
    std::unique_lock lock { m_mutex };

    while (!m_isCleared && m_heap.empty() && !m_currentlyWorkedOnLowerBounds.empty())
        m_cv.wait(lock);

    if (m_isCleared || (m_heap.empty() && m_currentlyWorkedOnLowerBounds.empty()))
        return std::nullopt;

    std::pop_heap(begin(m_heap), end(m_heap), m_comparer);

    std::optional result = std::move(m_heap.back());
    m_heap.pop_back();

    m_currentlyWorkedOnLowerBounds.emplace(threadId, result->LowerBound);

    return result;
}

void tsplp::BranchAndCutQueue::Push(
    double lowerBound, std::vector<Variable> fixedVariables0, std::vector<Variable> fixedVariables1)
{
    bool needsNotify = false;

    {
        std::unique_lock lock { m_mutex };

        if (m_isCleared)
            return;

        needsNotify = m_heap.empty();

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
    }

    if (needsNotify)
    {
        m_cv.notify_one();
        m_cv.notify_one();
    }
}
