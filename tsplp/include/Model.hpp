#pragma once

#include "Status.hpp"
#include "Variable.hpp"

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
        std::unique_ptr<ClpSimplex> m_spSimplexModel;
        std::vector<Variable> m_variables;

    public:
        explicit Model(size_t numberOfBinaryVariables);
        ~Model();

        Model(const Model& other);
        Model(Model&& other) noexcept;

        Model& operator=(Model other);

        friend void swap(Model& m1, Model& m2) noexcept;

        const std::vector<Variable>& GetVariables() const { return m_variables; }
        void SetObjective(const LinearVariableComposition& objective);
        void AddConstraints(std::span<const LinearConstraint> constraints);
        Status Solve();
    };

    void swap(Model& m1, Model& m2) noexcept;
}