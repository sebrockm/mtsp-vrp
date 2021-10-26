#pragma once

#include <memory>
#include <span>

class ClpSimplex;

namespace tsplp
{
    class Variable;
    class LinearVariableComposition;
    class LinearConstraint;
    enum class Status;

    class Model
    {
    private:
        std::unique_ptr<ClpSimplex> m_spSimplexModel;

    public:
        explicit Model(size_t numberOfBinaryVariables);
        std::span<const Variable> GetVariables() const;
        void SetObjective(const LinearVariableComposition&);
        void AddConstraints(std::span<const LinearConstraint>);
        Status Solve();
    };
}