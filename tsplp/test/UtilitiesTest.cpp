#include "Utilities.hpp"

#include <catch2/catch.hpp>

#include <thread>

using namespace std::chrono_literals;

TEST_CASE("Stop watch ", "[StopWatch]")
{
    const tsplp::utilities::StopWatch watch;
    std::this_thread::sleep_for(10ms);
    const auto elapsed = watch.GetTime();

    CHECK(elapsed.count() >= 10);
}
