#pragma once

#include <cstddef>

namespace tsplp
{
class Model;

class Variable
{
private:
    size_t m_id;

public:
    Variable() = default;
    explicit Variable(size_t id);

    double GetUpperBound(const Model& model) const;
    double GetLowerBound(const Model& model) const;

    void SetUpperBound(double upperBound, Model& model) const;
    void SetLowerBound(double lowerBound, Model& model) const;

    double GetObjectiveValue(const Model& model) const;
    double GetReducedCosts(const Model& model) const;
    size_t GetId() const;

    void Fix(double value, Model& model) const;
    void Unfix(Model& model) const;
};

struct VariableLess
{
    bool operator()(const Variable& lhs, const Variable& rhs) const;
};
}