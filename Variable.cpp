#include "Variable.hpp"

bool tsplp::VariableLess::operator()(const Variable& lhs, const Variable& rhs) const
{
    return lhs.GetId() < rhs.GetId();
}
