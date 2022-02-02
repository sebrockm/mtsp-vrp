#include "BranchAndCutQueue.hpp"

tsplp::BranchAndCutQueue::BranchAndCutQueue()
{
    m_heap.emplace_back();
}

void tsplp::BranchAndCutQueue::ClearAll()
{
    {
        std::unique_lock lock{ m_mutex };
        m_isCleared = true;
    }

    m_cv.notify_all();
}

void tsplp::BranchAndCutQueue::NotifyNodeDone()
{
    bool needsNotify = false;

    {
        std::unique_lock lock{ m_mutex };
        assert(m_dataInProgress > 0);
        --m_dataInProgress;
        needsNotify = m_dataInProgress == 0;
    }

    if (needsNotify)
        m_cv.notify_all();
}

std::optional<double> tsplp::BranchAndCutQueue::GetLowerBound() const
{
    std::unique_lock lock{ m_mutex };

    if (m_heap.empty())
        return std::nullopt;
    
    return m_heap.front().LowerBound;
}

size_t tsplp::BranchAndCutQueue::GetSize() const
{
    std::unique_lock lock{ m_mutex };

    return m_heap.size();
}

std::optional<tsplp::SData> tsplp::BranchAndCutQueue::Pop()
{
    std::unique_lock lock{ m_mutex };

    while (!m_isCleared && m_heap.empty() && m_dataInProgress > 0)
        m_cv.wait(lock);

    if (m_isCleared || (m_heap.empty() && m_dataInProgress == 0))
        return std::nullopt;

    std::pop_heap(begin(m_heap), end(m_heap), m_comparer);
    
    std::optional result = std::move(m_heap.back());
    m_heap.pop_back();

    ++m_dataInProgress;

    return result;
}

void tsplp::BranchAndCutQueue::Push(double lowerBound, std::vector<Variable> fixedVariables0, std::vector<Variable> fixedVariables1)
{
    bool needsNotify = false;

    {
        std::unique_lock lock{ m_mutex };

        if (m_isCleared)
            return;

        needsNotify = m_heap.empty();

        m_heap.emplace_back(lowerBound, std::move(fixedVariables0), std::move(fixedVariables1));
        std::push_heap(begin(m_heap), end(m_heap), m_comparer);
    }

    if (needsNotify)
        m_cv.notify_one();
}

void tsplp::BranchAndCutQueue::PushBranch(double lowerBound, std::vector<Variable> fixedVariables0, std::vector<Variable> fixedVariables1, Variable branchingVariable)
{
    bool needsNotify = false;

    {
        std::unique_lock lock{ m_mutex };

        if (m_isCleared)
            return;

        needsNotify = m_heap.empty();

        auto copyFixedVariables0 = fixedVariables0;
        auto copyFixedVariables1 = fixedVariables1;

        fixedVariables0.push_back(branchingVariable);
        copyFixedVariables1.push_back(branchingVariable);

        m_heap.emplace_back(lowerBound, std::move(fixedVariables0), std::move(fixedVariables1));
        std::push_heap(begin(m_heap), end(m_heap), m_comparer);

        m_heap.emplace_back(lowerBound, std::move(copyFixedVariables0), std::move(copyFixedVariables1));
        std::push_heap(begin(m_heap), end(m_heap), m_comparer);
    }

    if (needsNotify)
    {
        m_cv.notify_one();
        m_cv.notify_one();
    }
}
