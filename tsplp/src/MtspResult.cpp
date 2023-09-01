#include "MtspResult.hpp"

namespace tsplp
{

MtspResult::Bounds MtspResult::GetBounds() const
{
    std::unique_lock lock { m_mutex };
    return Bounds { .Lower = m_lowerBound, .Upper = m_upperBound };
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

MtspResult::Bounds MtspResult::UpdateUpperBound(double newUpperBound, std::vector<std::vector<size_t>>&& newPaths)
{
    std::unique_lock lock { m_mutex };
    if (newUpperBound < m_upperBound)
    {
        m_paths = std::move(newPaths);
        m_upperBound = newUpperBound;
        m_lowerBound = std::min(m_lowerBound, m_upperBound);
    }

     return Bounds { .Lower = m_lowerBound, .Upper = m_upperBound };
}

MtspResult::Bounds MtspResult::UpdateLowerBound(double newLowerBound)
{
    std::unique_lock lock { m_mutex };
    if (newLowerBound >= m_lowerBound)
        m_lowerBound = std::min(newLowerBound, m_upperBound);

     return Bounds { .Lower = m_lowerBound, .Upper = m_upperBound };
}
}
