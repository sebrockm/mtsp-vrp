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
        std::span<const Variable> GetVariables() const;
        void SetObjective(const LinearVariableComposition& objective);
        void AddConstraints(std::span<const LinearConstraint> constraints);
        Status Solve();
    };
}