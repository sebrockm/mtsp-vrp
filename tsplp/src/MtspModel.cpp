#include "MtspModel.hpp"
#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"

#include <cmath>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

tsplp::MtspModel::MtspModel(std::vector<int> startPositions, std::vector<int> endPositions, std::vector<double> weights)
    : m_startPositions(xt::adapt(std::move(startPositions))),
    m_endPositions(xt::adapt(std::move(endPositions))),
    A(std::size(startPositions)),
    N(static_cast<size_t>(std::sqrt(std::size(weights)))),
    m_model(A * N * N),
    W(xt::adapt(std::move(weights), { N, N })),
    X(xt::adapt(m_model.GetVariables(), { A, N, N }))
{
    m_model.SetObjective(xt::sum(W * X)());

    std::vector<LinearConstraint> constraints;
    constraints.reserve(A * N + 3 * A + N * (N - 1) / 2);

    // don't use self referring arcs (entries on diagonal)
    for (size_t a = 0; a < A; ++a)
        for (size_t n = 0; n < N; ++n)
            constraints.emplace_back(X(a, n, n) == 0);

    // special inequalities for start and end nodes
    for (size_t a = 0; a < A; ++a)
    {
        // We write X + 0 instead of X to turn summed up type from Variable to LinearVariableComposition.
        // That is necessary because xtensor initializes the sum with a conversion from 0 to ResultType and we
        // don't provide a conversion from int to Variable, but we do provide one from int to LinearVariableCompositon.
        constraints.emplace_back(xt::sum(xt::view(X + 0, a, m_startPositions[a], xt::all()))() == 1); // arcs out of start nodes
        constraints.emplace_back(xt::sum(xt::view(X + 0, a, xt::all(), m_endPositions[a]))() == 1); // arcs into end nodes
        constraints.emplace_back(X(a, m_endPositions[a], m_startPositions[(a + 1) % A]) == 1); // artificial connections from end to next start
    }

    // inequalities to disallow cycles of length 2
    for (size_t u = 0; u < N; ++u)
        for (size_t v = u + 1; v < N; ++v)
            constraints.emplace_back((xt::sum(xt::view(X + 0, xt::all(), u, v)) + xt::sum(xt::view(X + 0, xt::all(), v, u)))() <= 1);

    m_model.AddConstraints(constraints);
}
