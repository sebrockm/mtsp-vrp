#pragma once

namespace tsplp
{
enum class Status
{
    Optimal,
    Infeasible,
    Unbounded,
    Timeout,
    Error
};
}
