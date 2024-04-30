import numpy as np
from perceptron import Perceptron


class LinearPerceptron(Perceptron):

    def activation_func(self, value):
        """Identity function"""
        return value
    
    def compute_deltas(self) -> np.array:
        output_errors = self.expected_value - self.get_outputs()
        deltas = self.learning_rate * output_errors.reshape(-1, 1) * self.data

        return deltas

    def compute_error(self):
        # """Mean Squared Error - MSE"""
        p = self.data.shape[0]
        output_errors = self.expected_value - self.get_outputs()
        return np.power(output_errors, 2).sum() / p

    def is_converged(self):
        expected_value_amplitude = np.max(self.expected_value) - np.min(self.expected_value)
        percentage_threshold = 10/100 #VARIABLE
        return self.compute_error() < percentage_threshold * expected_value_amplitude
