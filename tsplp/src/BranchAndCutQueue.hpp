#pragma once

#include "Variable.hpp"
#include "TimedMutex.hpp"

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
    mutable TimedMutex m_mutex {"BnCq mutex"};
    TimedCV m_cv {"BnCq cv"};

public:
    BranchAndCutQueue();

public:
    void ClearAll();
    void NotifyNodeDone(size_t threadId);
    [[nodiscard]] std::optional<double> GetLowerBound() const;
    void UpdateCurrentLowerBound(size_t threadId, double currentLowerBound);
    [[nodiscard]] size_t GetSize() const;
    [[nodiscard]] size_t GetWorkedOnSize() const;
    std::optional<SData> Pop(size_t threadId);
    void Push(
        double lowerBound, std::vector<Variable> fixedVariables0,
        std::vector<Variable> fixedVariables1);
    void PushBranch(
        double lowerBound, std::vector<Variable> fixedVariables0,
        std::vector<Variable> fixedVariables1, Variable branchingVariable,
        std::vector<Variable> recursivelyFixed0);
};
}
