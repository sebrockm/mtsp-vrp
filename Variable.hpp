#pragma once

#include <span>

class ClpSimplex;

namespace tsplp
{
    class Variable
    {
    public:
        explicit Variable(ClpSimplex& model, int id);
        double GetUpperBound() const;
        double GetLowerBound() const;
        void SetUpperBound(double upperBound);
        void SetLowerBound(double lowerBound);
        int GetId() const;
    };

    class Variables
    {
    private:
        ClpSimplex* m_pModel;

    public:
        explicit Variables(ClpSimplex& model);
        size_t GetSize() const;
        Variable operator[](int id) const;
        std::span<const double> GetObjectiveValues() const;
    };

    struct VariableLess
    {
        bool operator()(const Variable& lhs, const Variable& rhs) const;
    };
}