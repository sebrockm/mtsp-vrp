#pragma once

#include "Status.hpp"
#include "Variable.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

class ClpSimplex;

namespace tsplp
{
class LinearVariableComposition;
class LinearConstraint;

class Model
{
    friend class Variable;

private:
    std::unique_ptr<ClpSimplex> m_spSimplexModel;
    std::unique_ptr<std::mutex> m_spModelMutex;
    std::vector<Variable> m_variables;
    size_t m_numberOfBinaryVariables;

public:
    explicit Model(size_t numberOfBinaryVariables);
    ~Model() noexcept;

    Model(const Model& other);
    Model(Model&& other) noexcept;

    Model& operator=(const Model&) = delete;
    Model& operator=(Model&&) = delete;
    Model& operator=(Model other);

    friend void swap(Model& m1, Model& m2) noexcept;

    [[nodiscard]] std::span<const Variable> GetBinaryVariables() const;
    void SetObjective(const LinearVariableComposition& objective);
    template <typename RandIterator>
    void AddConstraints(RandIterator first, RandIterator last);
    Variable AddVariable(double lowerBound, double upperBound);
    Status Solve(std::chrono::steady_clock::time_point endTime);
};

void swap(Model& m1, Model& m2) noexcept;
}
