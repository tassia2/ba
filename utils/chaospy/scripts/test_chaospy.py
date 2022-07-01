import numpy
import chaospy
import matplotlib.pyplot as pyplot

alpha = chaospy.Normal(1.5, 0.2)
beta = chaospy.Uniform(0.1, 0.2)
joint = chaospy.J(alpha, beta)
coordinates = numpy.linspace(0, 10, 1000)


def model_solver(parameters):
    """
    Simple ordinary differential equation solver.

    Args:
        parameters (numpy.ndarray):
            Hyper-parameters defining the model initial
            conditions alpha and growth rate beta.
            Assumed to have ``len(parameters) == 2``.

    Returns:
        (numpy.ndarray):
            Solution to the equation.
            Same shape as ``coordinates``.
    """
    alpha, beta = parameters
    return alpha*numpy.e**-(coordinates*beta)

gauss_quads = [
    chaospy.generate_quadrature(order, joint, rule="gaussian")
    for order in range(1, 8)
]

gauss_evals = [
    numpy.array([model_solver(node) for node in nodes.T])
    for nodes, weights in gauss_quads
]

expansions = [chaospy.generate_expansion(order, joint)
              for order in range(1, 10)]
expansions[0].round(10)


gauss_model_approx = [
    chaospy.fit_quadrature(expansion, nodes, weights, evals)
    for expansion, (nodes, weights), evals in zip(expansions, gauss_quads, gauss_evals)
]

model_approx = gauss_model_approx[4]
nodes, _ = gauss_quads[4]
evals = model_approx(*nodes)

pyplot.subplot(111)
pyplot.plot(coordinates, evals, alpha=0.3)
pyplot.title("Gaussian")
pyplot.show()


uniform = chaospy.Uniform(0, 4)
# ex = chaospy.E(uniform)
# print(ex)

# q0 = chaospy.variable()
# ex = chaospy.E(q0**3-1, uniform)
dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
ex = chaospy.E(dist)
print(ex)

expected = chaospy.E(gauss_model_approx[-2], joint)
std = chaospy.Std(gauss_model_approx[-2], joint)

expected[:4].round(4), std[:4].round(4)



pyplot.rc("figure", figsize=[6, 4])

pyplot.xlabel("coordinates")
pyplot.ylabel("model approximation")
pyplot.fill_between(
    coordinates, expected-2*std, expected+2*std, alpha=0.3)
pyplot.plot(coordinates, expected)

pyplot.show()

