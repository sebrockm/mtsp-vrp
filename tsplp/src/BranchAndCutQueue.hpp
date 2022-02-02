#pragma once

#include "Variable.hpp"

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

namespace tsplp
{
    struct SData
    {
        double LowerBound = -std::numeric_limits<double>::max();
        std::vector<Variable> FixedVariables0{};
        std::vector<Variable> FixedVariables1{};
        bool operator>(SData const& sd) const { return LowerBound > sd.LowerBound; }
    };

    class BranchAndCutQueue
    {
    private:
        std::vector<SData> m_heap{};
        std::greater<> m_comparer{};
        size_t m_dataInProgress = 0;
        bool m_isCleared = false;
        mutable std::mutex m_mutex;
        std::condition_variable m_cv;

    public:
        BranchAndCutQueue();

    public:
        void ClearAll();
        void NotifyNodeDone();
        std::optional<double> GetLowerBound() const;
        size_t GetSize() const;
        std::optional<SData> Pop();
        void Push(double lowerBound, std::vector<Variable> fixedVariables0, std::vector<Variable> fixedVariables1);
        void PushBranch(double lowerBound, std::vector<Variable> fixedVariables0, std::vector<Variable> fixedVariables1, Variable branchingVariable);
    };
}
