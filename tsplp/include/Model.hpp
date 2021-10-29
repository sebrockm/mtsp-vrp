#pragma once

#include "Status.hpp"

#include <memory>
#include <span>

class ClpSimplex;

namespace tsplp
{
    class Variable;
    class LinearVariableComposition;
    class LinearConstraint;

    class Model
    {
    private:
        std::unique_ptr<ClpSimplex> m_spSimplexModel;

    public:
        explicit Model(size_t numberOfBinaryVariables);

        size_t GetNumberOfVariables() const;
        Variable GetVariable(int id) const;

        void SetObjective(const LinearVariableComposition& objective);
        void AddConstraints(std::span<const LinearConstraint> constraints);
        Status Solve();
    };
}