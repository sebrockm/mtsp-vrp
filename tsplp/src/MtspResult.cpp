#include "MtspResult.hpp"

namespace tsplp
{

double MtspResult::GetLowerBound() const
{
    std::unique_lock lock { m_mutex };
    return m_lowerBound;
}

double MtspResult::GetUpperBound() const
{
    std::unique_lock lock { m_mutex };
    return m_upperBound;
}

bool MtspResult::HaveBoundsCrossed() const
{
    std::unique_lock lock { m_mutex };
    return m_lowerBound >= m_upperBound;
}

bool MtspResult::IsTimeoutHit() const
{
    std::unique_lock lock { m_mutex };
    return m_isTimeoutHit;
}

void MtspResult::SetTimeoutHit()
{
    std::unique_lock lock { m_mutex };
    m_isTimeoutHit = true;
}

void MtspResult::UpdateUpperBound(double newUpperBound, std::vector<std::vector<size_t>>&& newPaths)
{
    std::unique_lock lock { m_mutex };
    if (newUpperBound < m_upperBound)
    {
        m_paths = std::move(newPaths);
        m_upperBound = newUpperBound;
    }
}

void MtspResult::UpdateLowerBound(double newLowerBound)
{
    std::unique_lock lock { m_mutex };
    m_lowerBound = std::min(newLowerBound, m_upperBound);
}
}
