#pragma once

#include <span>

namespace tsplp
{
    class Variable;
    class LinearVariableComposition;
    class LinearConstraint;
    enum class Status;

    class Model
    {
    public:
        explicit Model(size_t numberOfBinaryVariables);
        std::span<const Variable> GetVariables() const;
        void SetObjective(const LinearVariableComposition&);
        void AddConstraints(std::span<const LinearConstraint>);
        Status Solve();
    };
}