#pragma once

#include <cstddef>

class ClpSimplex;

namespace tsplp
{
    class Variable
    {
    private:
        ClpSimplex* m_pModel;
        size_t m_id;

    public:
        Variable() = default;
        explicit Variable(ClpSimplex& model, size_t id);
        double GetUpperBound() const;
        double GetLowerBound() const;
        void SetUpperBound(double upperBound);
        void SetLowerBound(double lowerBound);
        double GetObjectiveValue() const;
        size_t GetId() const;
    };

    struct VariableLess
    {
        bool operator()(const Variable& lhs, const Variable& rhs) const;
    };
}