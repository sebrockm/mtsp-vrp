#pragma once

#include "Status.hpp"
#include "Variable.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

class ClpSimplex;

namespace tsplp
{
class LinearVariableComposition;
class LinearConstraint;

class Model // NOLINT(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
{
    friend class Variable;

private:
    std::unique_ptr<ClpSimplex> m_spSimplexModel;
    std::unique_ptr<std::mutex> m_spModelMutex;
    std::vector<Variable> m_variables;

public:
    Model() = default;
    explicit Model(size_t numberOfBinaryVariables);
    ~Model() noexcept;

    Model(const Model& other);
    Model(Model&& other) noexcept;

    Model& operator=(Model other);

    friend void swap(Model& m1, Model& m2) noexcept;

    [[nodiscard]] const std::vector<Variable>& GetVariables() const { return m_variables; }
    void SetObjective(const LinearVariableComposition& objective);
    template <typename RandIterator>
    void AddConstraints(RandIterator first, RandIterator last);
    Status Solve(std::chrono::steady_clock::time_point endTime);
};

void swap(Model& m1, Model& m2) noexcept;
}
