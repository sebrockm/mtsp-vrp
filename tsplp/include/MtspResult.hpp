#pragma once

#include <limits>
#include <mutex>
#include <vector>

namespace tsplp
{

class MtspResult
{
    struct Bounds
    {
        double Lower;
        double Upper;
    };

private:
    std::vector<std::vector<size_t>> m_paths {};
    double m_lowerBound = -std::numeric_limits<double>::max();
    double m_upperBound = std::numeric_limits<double>::max();
    bool m_isTimeoutHit = false;
    mutable std::mutex m_mutex;

public:
    [[nodiscard]] Bounds GetBounds() const;
    [[nodiscard]] bool IsTimeoutHit() const;
    [[nodiscard]] const auto& GetPaths() const { return m_paths; }

    void SetTimeoutHit();
    Bounds UpdateUpperBound(double newUpperBound, std::vector<std::vector<size_t>>&& newPaths);
    Bounds UpdateLowerBound(double newLowerBound);

    void Print() const;
};
}
