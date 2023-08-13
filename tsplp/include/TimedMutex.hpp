#pragma once

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>

class TimedMutex
{
private:
    std::mutex m_mutex {};
    std::chrono::nanoseconds m_waittime { 0 };
    std::string m_name;

public:
    TimedMutex(std::string name)
        : m_name(std::move(name))
    {
    }
    TimedMutex(const TimedMutex&) = delete;
    TimedMutex(TimedMutex&&) = delete;
    TimedMutex& operator=(const TimedMutex&) = delete;
    TimedMutex& operator=(TimedMutex&&) = delete;

    ~TimedMutex()
    {
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(m_waittime)
                  << " waited on Mutex " << m_name << std::endl;
    }

    void lock()
    {
        const auto start = std::chrono::system_clock::now();
        m_mutex.lock();
        const auto end = std::chrono::system_clock::now();

        m_waittime += end - start;
    }

    void unlock() { m_mutex.unlock(); }

    bool try_lock() { return m_mutex.try_lock(); }
};

class TimedCV
{
private:
    std::condition_variable_any m_cv {};
    std::chrono::nanoseconds m_waittime { 0 };
    std::string m_name;

public:
    TimedCV(std::string name)
        : m_name(std::move(name))
    {
    }
    TimedCV(const TimedCV&) = delete;
    TimedCV(TimedCV&&) = delete;
    TimedCV& operator=(const TimedCV&) = delete;
    TimedCV& operator=(TimedCV&&) = delete;

    ~TimedCV()
    {
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(m_waittime)
                  << " waited on CV " << m_name << std::endl;
    }

    void notify_one() noexcept { m_cv.notify_one(); }
    void notify_all() noexcept { m_cv.notify_all(); }

    void wait(std::unique_lock<TimedMutex>& lock)
    {
        const auto start = std::chrono::system_clock::now();
        m_cv.wait(lock);
        const auto end = std::chrono::system_clock::now();

        m_waittime += end - start;
    }
};