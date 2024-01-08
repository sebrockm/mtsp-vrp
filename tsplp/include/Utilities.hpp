#pragma once

#include <chrono>

namespace tsplp::utilities
{
class StopWatch
{
private:
    std::chrono::steady_clock::time_point start_time_;

public:
    StopWatch() noexcept;

    std::chrono::milliseconds GetTime() const noexcept;
};
}
