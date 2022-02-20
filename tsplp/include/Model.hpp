#pragma once

#include "Status.hpp"
#include "Variable.hpp"

#include <chrono>
#include <memory>
#include <span>
#include <vector>

class ClpSimplex;

namespace tsplp
{
    class LinearVariableComposition;
    class LinearConstraint;

    class Model
    {
    private:
        std::shared_ptr<ClpSimplex> m_spSimplexModel;
        std::vector<Variable> m_variables;

    public:
        explicit Model(size_t numberOfBinaryVariables);
        ~Model() noexcept;

        const std::vector<Variable>& GetVariables() const { return m_variables; }
        void SetObjective(const LinearVariableComposition& objective);
        void AddConstraints(std::span<const LinearConstraint> constraints);
        Status Solve(std::chrono::milliseconds timeout);
    };
}