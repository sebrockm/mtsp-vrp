#pragma once

namespace tsplp
{
    class Model;

    class Variable
    {
    public:
        explicit Variable(Model& model);
        double GetObjectiveValue() const;
        double GetUpperBound() const;
        double GetLowerBound() const;
        void SetUpperBound(double upperBound);
        void SetLowerBound(double lowerBound);
        int GetId() const;
    };



    struct VariableLess
    {
        bool operator()(const Variable& lhs, const Variable& rhs) const
        {
            return lhs.GetId() < rhs.GetId();
        }
    };
}