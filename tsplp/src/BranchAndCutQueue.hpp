#pragma once

#include "Variable.hpp"

#include <condition_variable>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tsplp
{
struct SData
{
    double LowerBound = -std::numeric_limits<double>::max();
    std::vector<Variable> FixedVariables0 {};
    std::vector<Variable> FixedVariables1 {};
    bool operator>(SData const& sd) const { return LowerBound > sd.LowerBound; }
};

class BranchAndCutQueue
{
private:
    std::vector<SData> m_heap {};
    std::greater<> m_comparer {};
    std::unordered_map<size_t, double> m_currentlyWorkedOnLowerBounds;
    bool m_isCleared = false;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;

public:
    BranchAndCutQueue();

public:
    [[nodiscard]] std::optional<double> GetLowerBound() const;
    [[nodiscard]] std::optional<SData> Pop(size_t threadId);

    void ClearAll();
    void NotifyNodeDone(size_t threadId);
    void UpdateCurrentLowerBound(size_t threadId, double currentLowerBound);
    void Push(
        double lowerBound, std::vector<Variable> fixedVariables0,
        std::vector<Variable> fixedVariables1);
    void PushBranch(
        double lowerBound, std::vector<Variable> fixedVariables0,
        std::vector<Variable> fixedVariables1, Variable branchingVariable,
        std::vector<Variable> recursivelyFixed0);
};
}
