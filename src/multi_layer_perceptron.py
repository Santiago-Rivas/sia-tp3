import numpy as np
from src.perceptron import Perceptron


def MultiLayerPerceptron(Perceptron):

    def __init__(
            self,
            learning_rate: float,
            inputs: np.array,
            hidden_nodes: int,
            output_nodes: int,
            expected_outputs: np.array
    ):
        self.learning_rate = learning_rate
        self.X = np.insert(inputs, 0,1,axis=1)

        expected_range = (np.min(expected_outputs),np.max(expected_outputs))
        self.Y = feature_scaling(expected_outputs, expected_range,(0,1))

        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.M = self.X.shapep[1]

        self.weights = [
            np.random.randn(self.hidden_nodes, self.M),
            np.random.randn(self.output_nodes, self.hidden_nodes + 1)
        ]

        self.previous_deltas = [np.zeros(self.weight[0].shape), np.zeros(self.weights[1].shape)]



    def activation_func(self, value):
        return 1 / (1 + np.exp(-value))
    
    def activation_func_derivate(self,value):
        activation_function = self.activation_func(value)
        return activation_function * (1 - activation_function)
    
    def predict(self, X):
        X = np.insert(X,0,1,axis=1)
        _, _, _, output = self.forward_propagation(X)
        return output
    
    def forward_propagation(self, X):
        h1 = self.weights[0].dot(X,T)
        # Hidden layer output
        V1 = self.activation_func(h1)  # (hidden_nodes, N)
        # Add bias to hidden layer output
        V1 = np.insert(V1, 0, 1, axis=0) # (hidden_nodes + 1, N)

        h2 = self.weights[1].dot(V1)# (output_nodes, hidden_nodes + 1) x (hidden_nodes + 1, N) = (output_nodes, N)
        # Output layer output
        o = self.activation_func(h2)
        return h1, V1, h2, o

    def backward_propagation(self, h1, V1, h2, O):
        # Update output layer weights
        output_errors = self.Y.T - O                              # (output_nodes, N) - (output_nodes, N) = (output_nodes, N)
        dO = output_errors * self.activation_func_derivative(h2)  # (output_nodes, N) * (output_nodes, N) = (output_nodes, N), multiply element by element
        dW = self.learning_rate * dO.dot(V1.T)                    # (output_nodes, N) x (N, hidden_nodes + 1) = (output_nodes, hidden_nodes + 1)
        
        # Update hidden layer weights
        output_layer_delta_sum = dO.T.dot(self.weights[1][:, 1:])        # (N, output_nodes) x (output_nodes, hidden_nodes) = (N, hidden_nodes) . Don't use the bias term in the calculation
        dV1 = output_layer_delta_sum.T * self.activation_func_derivative(h1) # (hidden_nodes, N) * (hidden_nodes, N) = (hidden_nodes, N)
        dw = self.learning_rate * dV1.dot(self.X)                 # (hidden_nodes, N) x (N, M) =  (hidden_nodes, M)

        if settings.optimization.active and settings.optimization.method == "momentum":
            aux_dW = dW.copy()
            aux_dw = dw.copy()
            dw += settings.optimization.momentum_rate * self.previous_deltas[0]
            dW += settings.optimization.momentum_rate * self.previous_deltas[1]
            self.previous_deltas = [aux_dw, aux_dW]

        self.weights[1] += dW
        self.weights[0] += dw

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            h1, V1, h2, O = self.feed_forward(self.X)

            if self.is_converged(O):
                break

            if epoch % 1000 == 0 and settings.verbose:
                print(f"{epoch=} ; output={O} ; error={self.get_error(O)}")

            self.backward_propagation(h1, V1, h2, O)

        return O, epoch, self.is_converged(O)

    def get_scaled_outputs(self):
        return
    
    def compute_error(self):
        p = self.X.shape[0]
        output_errors = self.Y.T - O  # (output_nodes, N) - (output_nodes, N) = (output_nodes, N)
        return np.power(output_errors, 2).sum() / p
    
    def compute_deltas(self, indexes: [int]):
        return
    
    def is_converged(self):
        expected_outputs_amplitude = 1 - 0  # amplitude of the expected output values (scaled to logistic function range)
        percentage_threshold = settings.multilayer_perceptron.convergence_threshold / 100
        return self.get_error(O) < percentage_threshold * expected_outputs_amplitude
