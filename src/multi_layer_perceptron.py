import numpy as np
from src.perceptron import Perceptron
from src.utils import feature_scaling
from typing import Callable
import src.functions as fun


class MultiLayerPerceptron(Perceptron):

    def __init__(
            self,
            inputs: np.array,
            expected_outputs: np.array,
            hidden_nodes_dimensions: [int],
            output_nodes_dimension: int,
            beta: float = 0.9,
            learning_rate: float = 0.01,
            sigmoid_func: Callable[[float, ...], float] = fun.sigmoid_exp,
            sigmoid_func_img: tuple[float, float] = (0, 1),
            sigmoid_func_derivative: Callable[[
                float, ...], float] = fun.sigmoid_exp_derivative,
    ):
        self.learning_rate = learning_rate

        self.sigmoid_func = sigmoid_func
        self.sigmoid_func_img = sigmoid_func_img
        self.sigmoid_func_derivative = sigmoid_func_derivative
        self.beta = beta

        self.inputs = np.insert(inputs, 0, 1, axis=1)

        expected_range = (np.min(expected_outputs), np.max(expected_outputs))
        self.scaled_expected_outputs = feature_scaling(expected_outputs, expected_range, (0, 1))
        self.expected_outputs = expected_outputs

        self.output_nodes_dimension = output_nodes_dimension
        self.hidden_nodes_dimensions = hidden_nodes_dimensions

        self.weights = []
        row = np.random.rand(len(self.inputs[0]), hidden_nodes_dimensions[0])
        self.weights.append(row)

        # Generate random numbers for each row
        for i in range(1, len(hidden_nodes_dimensions)):
            row = np.random.rand(hidden_nodes_dimensions[i - 1], hidden_nodes_dimensions[i])
            self.weights.append(row)

        self.weights.append(np.random.rand(hidden_nodes_dimensions[-1], output_nodes_dimension))

        self.M = self.inputs.shape[1]

        self.min_weights = self.weights

        self.previous_deltas = [
            np.zeros(self.weights[0].shape), np.zeros(self.weights[1].shape)]

    def activation_func(self, value):
        return self.activation_func(value, self.beta)

    def activation_func_derivate(self, value):
        return self.activation_func_derivate(value, self.beta)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        _, _, _, output = self.forward_propagation(X)
        return output[-1]

    def forward_propagation(self, X: [[int]]):
        Vs = []
        Vs.append(np.array(X))

        for i in range(len(self.weights)):
            excitations = np.dot(Vs[i], self.weights[i])
            output = np.vectorize(
                self.sigmoid_func)(excitations, self.beta)
            Vs.append(output)
        return Vs

    def backward_propagation(self, h1, V1, h2, O):
        # Update output layer weights
        # (output_nodes, N) - (output_nodes, N) = (output_nodes, N)
        output_errors = self.Y.T - O
        # (output_nodes, N) * (output_nodes, N) = (output_nodes, N), multiply element by element
        dO = output_errors * self.activation_func_derivative(h2)
        # (output_nodes, N) x (N, hidden_nodes + 1) = (output_nodes, hidden_nodes + 1)
        dW = self.learning_rate * dO.dot(V1.T)

        # Update hidden layer weights
        # (N, output_nodes) x (output_nodes, hidden_nodes) = (N, hidden_nodes) . Don't use the bias term in the calculation
        output_layer_delta_sum = dO.T.dot(self.weights[1][:, 1:])
        # (hidden_nodes, N) * (hidden_nodes, N) = (hidden_nodes, N)
        dV1 = output_layer_delta_sum.T * self.activation_func_derivative(h1)
        # (hidden_nodes, N) x (N, M) =  (hidden_nodes, M)
        dw = self.learning_rate * dV1.dot(self.X)

        self.weights[1] += dW
        self.weights[0] += dw

    def get_indexes(self):
        # u = random.randint(0, len(self.data) - 1)
        u = np.random.randint(0, len(self.inputs), 2)
        return u

    def train(self, max_epochs: int = 1000):
        for epoch in range(max_epochs):
            u = self.get_indexes()
            Vs = self.forward_propagation(self.inputs[u])
            print(Vs[-1])
            exit()
            # TODO: finish method

            if self.is_converged(Vs[-1]):
                break

            if epoch % 1000 == 0:
                print(f"{epoch=} ; output={Vs[-1]} ; error={self.get_error(Vs[-1])}")

            self.backward_propagation(h1, V1, h2, O)
        return O, epoch, self.is_converged(O)

    def get_scaled_outputs(self):
        return

    def compute_error(self, output, expected_outputs):
        p = self.inputs.shape[0]
        # (output_nodes, N) - (output_nodes, N) = (output_nodes, N)
        output_errors = self.scaled_expected_outputs - output
        return np.power(output_errors, 2).sum() / p

    def compute_deltas(self, indexes: [int]):
        return

    def is_converged(self, output):
        # amplitude of the expected output values (scaled to logistic function range)
        expected_outputs_amplitude = 1 - 0 # TODO: fix
        percentage_threshold = 0.0001
        return self.compute_error(output) < percentage_threshold * expected_outputs_amplitude
