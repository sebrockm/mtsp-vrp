#pragma once

#include <span>

namespace tsplp
{
    class Model;

    class Variable
    {
    public:
        explicit Variable(Model& model, int id);
        void SetUpperBound(double upperBound);
        void SetLowerBound(double lowerBound);
        int GetId() const;
    };

    class Variables
    {
    public:
        explicit Variables(Model& model);
        std::span<double> GetObjectiveValues() const;
        std::span<double> GetUpperBounds() const;
        std::span<double> GetLowerBounds() const;
        Variable operator[](int id) const;
    };

    struct VariableLess
    {
        bool operator()(const Variable& lhs, const Variable& rhs) const;
    };
}