#include "ConstraintDeque.hpp"

#include "Model.hpp"

#include <algorithm>

tsplp::ConstraintDeque::ConstraintDeque(size_t numberOfThreads)
    : m_readPositions(numberOfThreads)
{
}

void tsplp::ConstraintDeque::Push(LinearConstraint constraint)
{
    std::unique_lock lock { m_mutex };

    m_deque.push_back(std::move(constraint));
}

void tsplp::ConstraintDeque::PopToModel(size_t threadId, Model& model)
{
    const auto constraints = [this, threadId]() -> std::vector<LinearConstraint>
    {
        std::unique_lock lock { m_mutex };

        const auto prevReadPos = m_readPositions[threadId];
        m_readPositions[threadId] = std::ssize(m_deque);

        return { m_deque.cbegin() + prevReadPos, m_deque.cend() };
    }();

    model.AddConstraints(constraints.cbegin(), constraints.cend());

    // this step doesn't need to be performed often, so let only one thread do it
    if (threadId == 0)
    {
        std::unique_lock lock { m_mutex };

        const auto minReadPosition
            = *std::min_element(begin(m_readPositions), end(m_readPositions));
        m_deque.erase(m_deque.begin(), m_deque.begin() + minReadPosition);

        for (auto& position : m_readPositions)
            position -= minReadPosition;
    }
}
