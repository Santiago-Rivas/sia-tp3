from src.perceptron import Perceptron
import numpy as np


class StepPerceptron (Perceptron):

    def activation_func(self, value):
        return 1 if value >= 0 else -1

    def compute_error(self):
        return np.sum(abs(self.expected_value - self.get_outputs()))

    def compute_deltas(self, indexes):
        output_errors = self.expected_value[indexes] - self.get_range_outputs(indexes)
        print()
        print(self.expected_value[indexes])
        print(self.get_range_outputs(indexes))
        print(self.learning_rate)
        print("output_errors", output_errors)
        print("reshape", output_errors.reshape(-1, 1))
        print(self.data)
        deltas = self.learning_rate * output_errors.reshape(-1, 1) * self.data[indexes]
        return deltas

    def is_converged(self):
        # pasar a un config file los thresholds
        return self.compute_error() == 0
