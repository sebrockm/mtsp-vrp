#pragma once

#include "LinearConstraint.hpp"

#include <deque>
#include <mutex>
#include <vector>

namespace tsplp
{
class Model;

class ConstraintDeque
{
private:
    std::deque<LinearConstraint> m_deque;
    std::vector<ptrdiff_t> m_readPositions;
    std::mutex m_mutex;

public:
    ConstraintDeque(size_t numberOfThreads);

public:
    template <typename LinearConstraintIterator>
    void Push(LinearConstraintIterator first, LinearConstraintIterator last)
    {
        std::unique_lock lock { m_mutex };

        m_deque.insert(
            m_deque.end(), std::make_move_iterator(first), std::make_move_iterator(last));
    }

    void Push(LinearConstraint constraint);
    void PopToModel(size_t threadId, Model& model);
};
}
