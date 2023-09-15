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

    [[nodiscard]] double GetUpperBound(const Model& model) const;
    [[nodiscard]] double GetLowerBound(const Model& model) const;

    void SetUpperBound(double upperBound, Model& model) const;
    void SetLowerBound(double lowerBound, Model& model) const;

    [[nodiscard]] double GetObjectiveValue(const Model& model) const;
    [[nodiscard]] double GetReducedCosts(const Model& model) const;
    [[nodiscard]] size_t GetId() const;

    void Fix(double value, Model& model) const;
    void Unfix(Model& model) const;
};

inline bool operator==(Variable lhs, Variable rhs) { return lhs.GetId() == rhs.GetId(); }

struct VariableLess
{
    bool operator()(const Variable& lhs, const Variable& rhs) const;
};
}
