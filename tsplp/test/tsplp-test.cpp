#include "Model.hpp"
#include "Variable.hpp"
#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"
#include "Status.hpp"

#include <vector>
#include <catch2/catch.hpp>

TEST_CASE("3 variables, 3 constraints", "[lp]") 
{
    tsplp::Model model(3);

    auto x1 = model.GetVariables()[0];
    auto x2 = model.GetVariables()[1];
    auto x3 = model.GetVariables()[2];

    x1.SetLowerBound(0);
    x1.SetUpperBound(4);
    x2.SetLowerBound(-1);
    x2.SetUpperBound(1);
    x3.SetLowerBound(-std::numeric_limits<double>::max());
    x3.SetUpperBound(std::numeric_limits<double>::max());

    auto objective = x1 + 4 * x2 + 9 * x3 - 10;
    model.SetObjective(objective);

    auto c1 = x1 + x2      <=  5;
    auto c2 = x1 +      x3 >= 10;
    auto c3 =    - x2 + x3 ==  7;

    std::vector<tsplp::LinearConstraint> constraints{ c1, c2, c3 };

    model.AddConstraints(constraints);

    auto status = model.Solve();

    REQUIRE(status == tsplp::Status::Optimal);
    REQUIRE(c1.Evaluate());
    REQUIRE(c2.Evaluate());
    REQUIRE(c3.Evaluate());
    REQUIRE(objective.Evaluate() == Approx(44));
}