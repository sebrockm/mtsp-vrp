#pragma once

class ClpSimplex;

namespace tsplp
{
    class Variable
    {
    private:
        ClpSimplex* m_pModel;
        int m_id;

    public:
        Variable() = default;
        explicit Variable(ClpSimplex& model, int id);
        double GetUpperBound() const;
        double GetLowerBound() const;
        void SetUpperBound(double upperBound);
        void SetLowerBound(double lowerBound);
        double GetObjectiveValue() const;
        int GetId() const;
    };

    struct VariableLess
    {
        bool operator()(const Variable& lhs, const Variable& rhs) const;
    };
}