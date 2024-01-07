#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"
#include "Model.hpp"
#include "Status.hpp"
#include "Variable.hpp"

#include <catch2/catch.hpp>

#include <vector>

TEST_CASE("3 variables, 3 constraints", "[lp]")
{
    tsplp::Model model(3);

    REQUIRE(model.GetBinaryVariables().size() == 3);
    CHECK(model.GetNumberOfConstraints() == 0);

    auto x1 = model.GetBinaryVariables()[0];
    auto x2 = model.GetBinaryVariables()[1];
    auto x3 = model.GetBinaryVariables()[2];

    x1.SetLowerBound(0, model);
    x1.SetUpperBound(4, model);
    x2.SetLowerBound(-1, model);
    x2.SetUpperBound(1, model);
    x3.SetLowerBound(-std::numeric_limits<double>::max(), model);
    x3.SetUpperBound(std::numeric_limits<double>::max(), model);

    auto objective = x1 + 4 * x2 + 9 * x3 - 10;
    model.SetObjective(objective);

    CHECK(model.GetNumberOfConstraints() == 0);

    auto c1 = x1 + x2 <= 5;
    auto c2 = x1 + x3 >= 10;
    auto c3 = -x2 + x3 == 7;

    const std::vector<tsplp::LinearConstraint> constraints { c1, c2, c3 };

    model.AddConstraints(constraints.cbegin(), constraints.cend());

    CHECK(model.GetNumberOfConstraints() == 3);

    using namespace std::chrono_literals;
    auto status = model.Solve(std::chrono::steady_clock::now() + 10ms);

    CHECK(model.GetNumberOfConstraints() == 3);
    CHECK(status == tsplp::Status::Optimal);
    CHECK(c1.Evaluate(model));
    CHECK(c2.Evaluate(model));
    CHECK(c3.Evaluate(model));
    CHECK(objective.Evaluate(model) == Approx(44));
}
