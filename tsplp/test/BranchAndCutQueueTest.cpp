#include "BranchAndCutQueue.hpp"

#include <catch2/catch.hpp>

#include <atomic>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

TEST_CASE("NodeDoneNotifier calls function on destruction", "[NodeDoneNotifier]")
{
    int counter = 0;
    {
        tsplp::NodeDoneNotifier notifier { [&] { ++counter; } };
        CHECK(counter == 0);
    }
    CHECK(counter == 1);
}

TEST_CASE("NodeDoneNotifier can be moved", "[NodeDoneNotifier]")
{
    int counter = 0;
    {
        tsplp::NodeDoneNotifier notifier { [&] { ++counter; } };
        CHECK(counter == 0);

        auto notifier2 = std::move(notifier);
        CHECK(counter == 0);
    }
    CHECK(counter == 1);
}

TEST_CASE("NodeDoneNotifier can be moved inside of an optional tuple", "[NodeDoneNotifier]")
{
    int counter = 0;
    {
        auto optTupNotifier = std::invoke(
            [&]() -> std::optional<std::tuple<int, tsplp::NodeDoneNotifier>>
            {
                tsplp::NodeDoneNotifier notifier { [&] { ++counter; } };
                auto tup = std::make_tuple(0, std::move(notifier));
                return tup;
            });
        CHECK(counter == 0);

        const auto [i, notifier] = std::move(*optTupNotifier);
        CHECK(counter == 0);
    }
    CHECK(counter == 1);
}

TEST_CASE("BranchAndCutQueue constructor", "[BranchAndCutQueue]")
{
    CHECK_THROWS(tsplp::BranchAndCutQueue(0));
}

SCENARIO("BranchAndCutQueue usage", "[BranchAndCutQueue]")
{
    GIVEN("a BranchAndCutQueue with 1 thread")
    {
        tsplp::BranchAndCutQueue q(1);

        THEN("lower bound is initialized to -std::numeric_limits<double>::max()")
        {
            CHECK(q.GetLowerBound() == -std::numeric_limits<double>::max());
        }

        THEN("updating the current lower bound is an error (as no data is currently popped)")
        {
            CHECK_THROWS(q.UpdateCurrentLowerBound(0, 13));
        }

        THEN("Pop returns nullopt for thread 0") { CHECK(q.Pop(0) == std::nullopt); }

        THEN("Pop throws for threads >= 1")
        {
            CHECK_THROWS(q.Pop(1));
            CHECK_THROWS(q.Pop(10));
        }

        WHEN("data is pushed")
        {
            const std::vector<tsplp::Variable> fixed0
                = { tsplp::Variable { 1 }, tsplp::Variable { 2 } };
            const std::vector<tsplp::Variable> fixed1 = { tsplp::Variable { 3 } };
            const double lb = 12;
            q.Push(lb, fixed0, fixed1);

            THEN("lower bound updates accordingly") { CHECK(q.GetLowerBound() == lb); }

            THEN("pushing a worse lower bound than the current is an error")
            {
                CHECK_THROWS(q.Push(lb - 1, {}, {}));
                CHECK_THROWS(q.PushBranch(lb - 1, {}, {}, tsplp::Variable { 0 }, {}));
            }

            AND_WHEN("SData is popped")
            {
                auto data = q.Pop(0);

                THEN("popped data is identical to previously pushed")
                {
                    REQUIRE(data.has_value());

                    auto [sdata, notifier] = std::move(*data);

                    CHECK(sdata.LowerBound == 12);
                    CHECK(sdata.FixedVariables0 == fixed0);
                    CHECK(sdata.FixedVariables1 == fixed1);

                    AND_WHEN("improved data is pushed")
                    {
                        q.Push(sdata.LowerBound + 1, sdata.FixedVariables0, sdata.FixedVariables1);

                        THEN("lower bound stays untouched, for now")
                        {
                            CHECK(q.GetLowerBound() == lb);
                        }

                        AND_WHEN("notifier goes out of scope")
                        {
                            {
                                // move notifier to a variable that goes out of scope
                                const auto movedNotifier = std::move(notifier);
                            }

                            THEN("lower bound updates") { CHECK(q.GetLowerBound() == lb + 1); }
                        }
                    }
                }

                THEN("lower bound stays") { CHECK(q.GetLowerBound() == lb); }

                THEN("UpdateCurrentLowerBound with worse lb throws")
                {
                    CHECK_THROWS(q.UpdateCurrentLowerBound(0, lb - 1));
                    CHECK(q.GetLowerBound() == lb);
                }

                THEN("UpdateCurrentLowerBound with better lb changes lb")
                {
                    q.UpdateCurrentLowerBound(0, lb + 1);
                    CHECK(q.GetLowerBound() == lb + 1);
                }
            }

            AND_WHEN("ClearAll is called")
            {
                q.ClearAll();

                THEN("lower bound is -std::numeric_limits<double>::max()")
                {
                    CHECK(q.GetLowerBound() == -std::numeric_limits<double>::max());
                }

                THEN("Pop returns nullopt") { CHECK_FALSE(q.Pop(0).has_value()); }

                THEN("another ClearAll doesn't harm") { CHECK_NOTHROW(q.ClearAll()); }

                THEN("updating the current lower bound has no effect")
                {
                    CHECK_NOTHROW(q.UpdateCurrentLowerBound(0, 13));
                    CHECK(q.GetLowerBound() == -std::numeric_limits<double>::max());
                }

                THEN("Push has no effect")
                {
                    q.Push(lb + 1, {}, {});
                    CHECK_FALSE(q.Pop(0).has_value());
                    CHECK(q.GetLowerBound() == -std::numeric_limits<double>::max());
                }

                THEN("PushBranch has no effect")
                {
                    q.PushBranch(lb + 1, {}, {}, tsplp::Variable { 0 }, {});
                    CHECK_FALSE(q.Pop(0).has_value());
                    CHECK(q.GetLowerBound() == -std::numeric_limits<double>::max());
                }
            }
        }

        WHEN("a branching variable is pushed")
        {
            std::vector<tsplp::Variable> fixed0 = { tsplp::Variable { 1 }, tsplp::Variable { 2 } };
            tsplp::Variable branchingVar { 3 };
            std::vector<tsplp::Variable> fixed1 = { tsplp::Variable { 4 } };
            std::vector<tsplp::Variable> recFixed0 = { tsplp::Variable { 5 } };
            const double lb = 12;

            q.PushBranch(lb, fixed0, fixed1, branchingVar, recFixed0);

            THEN("lower bound updates accordingly") { CHECK(q.GetLowerBound() == lb); }

            THEN("we can pop exactly 2 times with correct data")
            {
                {
                    auto p = q.Pop(0);
                    REQUIRE(p.has_value());

                    const auto [top, n] = std::move(*p);

                    const std::vector<tsplp::Variable> top1ExpectedFixed0
                        = { tsplp::Variable { 1 }, tsplp::Variable { 2 }, branchingVar };

                    CHECK(top.LowerBound == lb);
                    CHECK(top.FixedVariables0 == top1ExpectedFixed0);
                    CHECK(top.FixedVariables1 == fixed1);
                }
                {
                    auto p = q.Pop(0);
                    REQUIRE(p.has_value());

                    const auto [top, n] = std::move(*p);

                    const std::vector<tsplp::Variable> top2ExpectedFixed0
                        = { tsplp::Variable { 1 }, tsplp::Variable { 2 }, tsplp::Variable { 5 } };

                    const std::vector<tsplp::Variable> top2ExpectedFixed1
                        = { tsplp::Variable { 4 }, branchingVar };

                    CHECK(top.LowerBound == lb);
                    CHECK(top.FixedVariables0 == top2ExpectedFixed0);
                    CHECK(top.FixedVariables1 == top2ExpectedFixed1);
                }

                CHECK_FALSE(q.Pop(0).has_value());
            }
        }

        WHEN("ClearAll is called")
        {
            q.ClearAll();

            THEN("lower bound is -std::numeric_limits<double>::max()")
            {
                CHECK(q.GetLowerBound() == -std::numeric_limits<double>::max());
            }

            THEN("Pop returns nullopt") { CHECK_FALSE(q.Pop(0).has_value()); }

            THEN("another ClearAll doesn't harm") { CHECK_NOTHROW(q.ClearAll()); }

            THEN("Push has no effect")
            {
                q.Push(1, {}, {});
                CHECK_FALSE(q.Pop(0).has_value());
                CHECK(q.GetLowerBound() == -std::numeric_limits<double>::max());
            }

            THEN("PushBranch has no effect")
            {
                q.PushBranch(1, {}, {}, tsplp::Variable { 0 }, {});
                CHECK_FALSE(q.Pop(0).has_value());
                CHECK(q.GetLowerBound() == -std::numeric_limits<double>::max());
            }
        }

        WHEN("multiple different lower bounds are pushed")
        {
            q.Push(1, {}, {});
            q.Push(3, {}, {});
            q.Push(5, {}, {});
            q.Push(9, {}, {});
            q.Push(1, {}, {});
            q.Push(3, {}, {});

            THEN("lower bound increases during popping")
            {
                CHECK(q.GetLowerBound() == 1);
                {
                    std::ignore = q.Pop(0);
                }
                CHECK(q.GetLowerBound() == 1);
                {
                    std::ignore = q.Pop(0);
                }
                CHECK(q.GetLowerBound() == 3);
                {
                    std::ignore = q.Pop(0);
                }
                CHECK(q.GetLowerBound() == 3);
                {
                    std::ignore = q.Pop(0);
                }
                CHECK(q.GetLowerBound() == 5);
                {
                    std::ignore = q.Pop(0);
                }
                CHECK(q.GetLowerBound() == 9);
                {
                    std::ignore = q.Pop(0);
                }
                CHECK(q.Pop(0) == std::nullopt);
            }
        }
    }

    GIVEN("a BranchAndCutQueue with 2 threads")
    {
        tsplp::BranchAndCutQueue q(2);

        THEN("lower bound is initialized to -std::numeric_limits<double>::max()")
        {
            CHECK(q.GetLowerBound() == -std::numeric_limits<double>::max());
        }

        THEN("updating the current lower bound is an error (as no data is currently popped)")
        {
            CHECK_THROWS(q.UpdateCurrentLowerBound(0, 13));
            CHECK_THROWS(q.UpdateCurrentLowerBound(1, 13));
        }

        THEN("Pop returns nullopt for both threads")
        {
            CHECK(q.Pop(0) == std::nullopt);
            CHECK(q.Pop(1) == std::nullopt);
        }

        THEN("Pop throws for threads >= 2")
        {
            CHECK_THROWS(q.Pop(2));
            CHECK_THROWS(q.Pop(20));
        }

        WHEN("data is pushed")
        {
            const std::vector<tsplp::Variable> fixed0
                = { tsplp::Variable { 1 }, tsplp::Variable { 2 } };
            const std::vector<tsplp::Variable> fixed1 = { tsplp::Variable { 3 } };
            const double lb = 12;
            q.Push(lb, fixed0, fixed1);

            THEN("lower bound updates accordingly") { CHECK(q.GetLowerBound() == lb); }

            AND_WHEN("two threads try popping")
            {
                auto data = q.Pop(0);

                std::atomic_bool t2_pop_should_succeed = false;
                std::atomic_bool has_t2_popped = false;
                std::atomic_bool t2_may_finish = false;
                std::thread t2(
                    [&]
                    {
                        auto data2 = q.Pop(1);
                        has_t2_popped = true;

                        CHECK(data2.has_value() == t2_pop_should_succeed);

                        if (data2.has_value())
                        {
                            // lb + 1 is pushed below
                            CHECK(std::get<0>(*data2).LowerBound == lb + 1);
                        }

                        while (!t2_may_finish)
                            ;
                    });

                THEN("thread 2 needs to wait") { CHECK_FALSE(has_t2_popped); }

                THEN("popped data is identical to previously pushed")
                {
                    REQUIRE(data.has_value());

                    const auto [sdata, notifier] = std::move(*data);

                    CHECK(sdata.LowerBound == lb);
                    CHECK(sdata.FixedVariables0 == fixed0);
                    CHECK(sdata.FixedVariables1 == fixed1);
                }

                THEN("lower bound stays") { CHECK(q.GetLowerBound() == lb); }

                AND_WHEN("another push is done")
                {
                    t2_pop_should_succeed = true;
                    q.Push(lb + 1, {}, {});
                    std::this_thread::sleep_for(100ms);

                    THEN("thread 2 stops waiting on pop") { CHECK(has_t2_popped); }

                    THEN("lower bound stays") { CHECK(q.GetLowerBound() == lb); }

                    AND_WHEN("first notifier goes out of scope")
                    {
                        {
                            const auto notifier = std::move(std::get<1>(*data));
                        }
                        THEN("lower bound from second push is active")
                        {
                            CHECK(q.GetLowerBound() == lb + 1);
                        }

                        AND_WHEN("second notifier goes out of scope")
                        {
                            t2_may_finish = true;
                            std::this_thread::sleep_for(100ms);

                            THEN("lower bound is -std::numeric_limits<double>::max()")
                            {
                                CHECK(q.GetLowerBound() == -std::numeric_limits<double>::max());
                            }
                        }
                    }

                    AND_WHEN("second notifier goes out of scope")
                    {
                        t2_may_finish = true;
                        std::this_thread::sleep_for(100ms);

                        THEN("lower bound from first push is active")
                        {
                            CHECK(q.GetLowerBound() == lb);
                        }

                        AND_WHEN("first notifier goes out of scope")
                        {
                            {
                                const auto notifier = std::move(std::get<1>(*data));
                            }

                            THEN("lower bound is -std::numeric_limits<double>::max()")
                            {
                                CHECK(q.GetLowerBound() == -std::numeric_limits<double>::max());
                            }
                        }
                    }
                }

                AND_WHEN("ClearAll is called")
                {
                    q.ClearAll();
                    std::this_thread::sleep_for(100ms);

                    THEN("thread 2 stops waiting on pop") { CHECK(has_t2_popped); }
                    THEN("lower bound is -std::numeric_limits<double>::max()")
                    {
                        CHECK(q.GetLowerBound() == -std::numeric_limits<double>::max());
                    }
                }

                {
                    t2_may_finish = true;
                    // notifier needs to go out of scope before thread can join
                    const auto notifier = std::move(std::get<1>(*data));
                }
                t2.join();
            }
        }
    }

    GIVEN("a BranchAndCutQueue with 10 threads")
    {
        tsplp::BranchAndCutQueue q(10);

        const auto setup = [&](bool doClearAll)
        {
            using namespace std::chrono_literals;

            q.Push(0, {}, {});
            std::mutex history_mutex;
            std::vector<double> history;
            const auto runner = [&](size_t threadId)
            {
                while (true)
                {
                    const auto node = q.Pop(threadId);
                    if (!node.has_value())
                        break;

                    const auto& [data, n] = *node;
                    const auto lb = data.LowerBound;
                    {
                        std::unique_lock lock { history_mutex };
                        history.push_back(lb);
                    }

                    std::this_thread::sleep_for(50ms);

                    if (static_cast<int>(lb) % 2 == 1)
                    {
                        // for odd lb's, push
                        q.Push(lb + 1, {}, {});
                    }
                    else if (lb < 8)
                    {
                        // for even lb's, push branch
                        q.PushBranch(lb + 1, {}, {}, tsplp::Variable { 0 }, {});
                    }
                    else if (doClearAll)
                    {
                        // clear at lb == 8
                        q.ClearAll();
                    }
                }
            };

            std::vector<std::thread> threads;
            for (size_t i = 0; i < 10; ++i)
                threads.emplace_back(runner, i);

            for (auto& t : threads)
                t.join();

            return history;
        };

        WHEN("they are pushing and popping data with a ClearAll call at lb==8")
        {
            const auto history = setup(true);

            THEN("the lower bound history is correct")
            {
                const std::vector<double> expected { 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5,
                                                     5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7,
                                                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8 };
                CHECK(history.size() >= expected.size());
                CHECK(history.back() == 8);
            }
        }

        WHEN("they are pushing and popping data without ClearAll")
        {
            const auto history = setup(false);

            THEN("the lower bound history is correct")
            {
                CHECK(history == std::vector<double> { 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                                                       5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                                       6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                                                       7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8,
                                                       8, 8, 8, 8, 8, 8, 8, 8, 8 });
            }
        }
    }
}
