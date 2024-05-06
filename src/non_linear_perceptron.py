import numpy as np
from src.perceptron import Perceptron
from src.utils import feature_scaling
from typing import Callable
import time


def sigmoid_tanh(x, beta: float):
    return np.tanh(beta * x)


def sigmoid_tanh_derivative(x, beta: float):
    return beta * (1 - sigmoid_tanh(x, beta)**2)


def sigmoid_exp(x, beta: float):
    return 1 / (1 + np.exp(-2 * beta * x))


def sigmoid_exp_derivative(x, beta: float):
    return 2 * beta * sigmoid_exp(x, beta) * (1 - sigmoid_exp(x, beta))


class NonLinearPerceptron(Perceptron):

    def __init__(
        self,
        data: np.array,
        expected_value: np.array,
        beta: float = 0.9,
        learning_rate: float = 15,
        sigmoid_func: Callable[[float, ...], float] = sigmoid_exp,
        sigmoid_func_img: tuple[float, float] = (0, 1),
        sigmoid_func_derivative: Callable[[
            float, ...], float] = sigmoid_exp_derivative,
        percentage_threshold=0.0001
    ):
        super().__init__(data, expected_value, learning_rate)

        self.expected_range = (np.min(self.expected_value),
                               np.max(self.expected_value))
        self.sigmoid_func = sigmoid_func
        self.sigmoid_func_img = sigmoid_func_img
        self.sigmoid_func_derivative = sigmoid_func_derivative
        self.scaled_expected_values = feature_scaling(
            self.expected_value, self.expected_range, self.sigmoid_func_img)
        self.beta = beta
        self.percentage_threshold = percentage_threshold
        self.data_len = len(self.weights)

    def activation_func(self, value):
        # time.sleep(0.1)
        # print("value", value)
        # print("pritn", self.sigmoid_func(value, self.beta))
        return self.sigmoid_func(value, self.beta)

    def get_scaled_outputs(self):
        outputs = self.get_outputs()
        scaled_outputs = [feature_scaling(
            o, self.sigmoid_func_img, self.expected_range) for o in outputs]
        return np.array(scaled_outputs)

    def get_scaled_outputs_range(self, indexes):
        outputs = self.get_range_outputs(indexes)
        scaled_outputs = [feature_scaling(
            o, self.sigmoid_func_img, self.expected_range) for o in outputs]
        return np.array(scaled_outputs)

    def compute_error(self):
        p = self.data.shape[0]
        output_errors = self.scaled_expected_values - self.get_outputs()
        return np.power(output_errors, 2).sum() / p

    def compute_deltas(self, indexes: [int]):
        output_errors = self.scaled_expected_values[indexes] - \
            self.get_range_outputs(indexes)

        excitations = np.dot(self.data[indexes], self.weights)
        derivatives = np.vectorize(
            self.sigmoid_func_derivative)(excitations, self.beta)

        deltas = np.zeros(self.data_len)
        deltas = self.learning_rate * \
            (output_errors * derivatives).reshape(-1, 1) * self.data[indexes]

        return deltas

    def is_converged(self):
        expected_values_amplitude = np.max(
            self.scaled_expected_values) - np.min(self.scaled_expected_values)
        return self.compute_error() < self.percentage_threshold * expected_values_amplitude

    def __str__(self) -> str:
        output = "Expected - Actual\n"

        for expected, actual in zip(self.expected_value, self.get_scaled_outputs()):
            output += f"{expected:<10} {actual}\n"

        output += f"\nWeights: {self.weights}"

        return output
