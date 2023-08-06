#pragma once

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
    mutable std::mutex m_mutex;

public:
    bool IsTimeoutHit = false;

public:
    [[nodiscard]] double GetLowerBound() const;
    [[nodiscard]] double GetUpperBound() const;
    [[nodiscard]] bool HaveBoundsCrossed() const;
    [[nodiscard]] const auto& GetPaths() const { return m_paths; }

    void UpdateUpperBound(double newUpperBound, std::vector<std::vector<size_t>>&& newPaths);
    void UpdateLowerBound(double newLowerBound);
};
}
