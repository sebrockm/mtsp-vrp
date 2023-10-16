#pragma once

#include "Variable.hpp"

#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace tsplp
{
struct SData
{
    double LowerBound = -std::numeric_limits<double>::max();
    std::vector<Variable> FixedVariables0 {};
    std::vector<Variable> FixedVariables1 {};
    bool IsResult = false;
    bool operator>(SData const& sd) const { return LowerBound > sd.LowerBound; }
};

class NodeDoneNotifier
{
private:
    std::function<void()> m_notifyNodeDone;

public:
    explicit NodeDoneNotifier(std::function<void()> notifyNodeDone)
        : m_notifyNodeDone(std::move(notifyNodeDone))
    {
    }
    NodeDoneNotifier(const NodeDoneNotifier&) = delete;
    NodeDoneNotifier(NodeDoneNotifier&& other) noexcept
        : m_notifyNodeDone(std::move(other.m_notifyNodeDone))
    {
        other.m_notifyNodeDone = nullptr;
    }
    NodeDoneNotifier& operator=(const NodeDoneNotifier&) = delete;
    NodeDoneNotifier& operator=(NodeDoneNotifier&& other) = delete;

    ~NodeDoneNotifier()
    {
        if (m_notifyNodeDone)
            m_notifyNodeDone();
    }
};

class BranchAndCutQueue
{
private:
    std::vector<SData> m_heap {};
    std::greater<> m_comparer {};
    std::vector<std::optional<double>> m_workedOnLowerBounds;
    size_t m_workedOnCount = 0;
    bool m_isCleared = false;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;

public:
    explicit BranchAndCutQueue(size_t threadCount);

public:
    [[nodiscard]] double GetLowerBound() const;
    [[nodiscard]] std::optional<std::tuple<SData, NodeDoneNotifier>> Pop(size_t threadId);

    void ClearAll();
    void UpdateCurrentLowerBound(size_t threadId, double currentLowerBound);
    void PushResult(double lowerBound);
    void Push(
        double lowerBound, std::vector<Variable> fixedVariables0,
        std::vector<Variable> fixedVariables1);
    void PushBranch(
        double lowerBound, std::vector<Variable> fixedVariables0,
        std::vector<Variable> fixedVariables1, Variable branchingVariable,
        std::vector<Variable> recursivelyFixed0);

private:
    void NotifyNodeDone(size_t threadId);
    double CalculateLowerBound() const;
};
}
