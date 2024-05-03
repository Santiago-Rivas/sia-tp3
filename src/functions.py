import numpy as np


def sigmoid_tanh(x, beta: float):
    return np.tanh(beta * x)


def sigmoid_tanh_derivative(x, beta: float):
    return beta * (1 - sigmoid_tanh(x, beta)**2)


def sigmoid_exp(x, beta: float):
    return 1 / (1 + np.exp(-2 * beta * x))


def sigmoid_exp_derivative(x, beta: float):
    return 2 * beta * sigmoid_exp(x, beta) * (1 - sigmoid_exp(x, beta))

