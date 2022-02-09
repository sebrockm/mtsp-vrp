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
        std::vector<size_t> m_readPositions;
        std::mutex m_mutex;

    public:
        ConstraintDeque(size_t numberOfThreads);

    public:
        void Push(LinearConstraint constraint);
        void PopToModel(size_t threadId, Model& model);
    };
}
