#include "ConstraintDeque.hpp"

#include "Model.hpp"

tsplp::ConstraintDeque::ConstraintDeque(size_t numberOfThreads)
    : m_readPositions(numberOfThreads)
{
}

void tsplp::ConstraintDeque::Push(LinearConstraint constraint)
{
    std::unique_lock lock{ m_mutex };

    m_deque.push_back(std::move(constraint));
}

void tsplp::ConstraintDeque::PopToModel(size_t threadId, Model& model)
{
    std::unique_lock lock{ m_mutex };

    model.AddConstraints(m_deque.cbegin() + m_readPositions[threadId], m_deque.cend());

    m_readPositions[threadId] = m_deque.size();

    const auto minReadPosition = *std::min_element(begin(m_readPositions), end(m_readPositions));
    m_deque.erase(m_deque.begin(), m_deque.begin() + minReadPosition);

    for (auto& position : m_readPositions)
        position -= minReadPosition;
}
