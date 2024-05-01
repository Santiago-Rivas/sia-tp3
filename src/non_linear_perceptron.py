import numpy as np
from src.perceptron import Perceptron
from src.utils import feature_scaling


class NonLinearPerceptron(Perceptron):

    def __init__(
        self,
        learning_rate,
        data,
        expected_value,
        sigmoid_func,
        sigmoid_func_img,
        sigmoid_func_derivative
    ):
        super().__init__(data, expected_value, learning_rate)

        self.expected_range = (np.min(self.expected_value),
                               np.max(self.expected_value))
        self.sigmoid_func = sigmoid_func
        self.sigmoid_func_img = sigmoid_func_img
        self.sigmoid_func_derivative = sigmoid_func_derivative
        self.scaled_expected_values = feature_scaling(
            self.expected_value, self.expected_range, self.sigmoid_func_img)

    def activation_func(self, value):
        return self.sigmoid_func(value)

    def get_scaled_outputs(self):
        outputs = self.get_outputs()
        scaled_outputs = [feature_scaling(
            o, self.sigmoid_func_img, self.expected_range) for o in outputs]

        return np.array(scaled_outputs)

    def compute_error(self):
        p = self.data.shape[0]
        output_errors = self.scaled_expected_values - self.get_outputs()
        return np.power(output_errors, 2).sum() / p

    def compute_deltas(self):
        output_errors = self.scaled_expected_values - self.get_outputs()

        excitations = np.dot(self.data, self.weights)
        derivatives = np.vectorize(self.sigmoid_func_derivative)(excitations)

        deltas = self.learning_rate * \
            (output_errors * derivatives).reshape(-1, 1) * self.data

        return deltas

    def is_converged(self):
        expected_values_amplitude = np.max(
            self.scaled_expected_values) - np.min(self.scaled_expected_values)
        percentage_threshold = 0.05 / 100
        return self.get_error() < percentage_threshold * expected_values_amplitude
