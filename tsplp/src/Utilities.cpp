#include "Utilities.hpp"

tsplp::utilities::StopWatch::StopWatch() noexcept
    : start_time_(std::chrono::steady_clock::now())
{
}

std::chrono::milliseconds tsplp::utilities::StopWatch::GetTime() const noexcept
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time_);
}
