#pragma once

#include "TimedMutex.hpp"

#include <limits>
#include <mutex>
#include <vector>

namespace tsplp
{

class MtspResult
{
private:
    std::vector<std::vector<size_t>> m_paths {};
    double m_lowerBound = -std::numeric_limits<double>::max();
    double m_upperBound = std::numeric_limits<double>::max();
    bool m_isTimeoutHit = false;
    mutable TimedMutex m_mutex;

public:
    MtspResult(std::string name)
        : m_mutex(std::move(name))
    {
    }
    [[nodiscard]] double GetLowerBound() const;
    [[nodiscard]] double GetUpperBound() const;
    [[nodiscard]] bool HaveBoundsCrossed() const;
    [[nodiscard]] bool IsTimeoutHit() const;
    [[nodiscard]] const auto& GetPaths() const { return m_paths; }

    void SetTimeoutHit();
    void UpdateUpperBound(double newUpperBound, std::vector<std::vector<size_t>>&& newPaths);
    void UpdateLowerBound(double newLowerBound);
};
}
