import numpy as np
from src.perceptron import Perceptron

def feature_scaling(value: float, from_int: tuple[float, float], to_int: tuple[float, float]) -> float:
    numerator = value - from_int[0]
    denominator = from_int[1] - from_int[0]
    return (numerator / denominator) * (to_int[1] - to_int[0]) + to_int[0]

class MultiLayerPerceptron():

    def __init__(
            self,
            learning_rate: float,
            inputs: np.array,
            hidden_nodes: int,
            hidden_layers:int,
            output_nodes: int,
            expected_outputs: np.array
    ):
        self.learning_rate = learning_rate
        self.X = np.insert(inputs, 0,1,axis=1)

        expected_range = (np.min(expected_outputs),np.max(expected_outputs))
        self.Y = feature_scaling(expected_outputs, expected_range,(0,1))

        self.hidden_nodes = hidden_nodes
        self.hidden_layers = hidden_layers
        self.output_nodes = output_nodes

        self.M = self.X.shape[1]

        # Initialize weights for each layer
        self.weights = [np.random.randn(self.hidden_nodes, self.M)]  # Input layer to first hidden layer
        
        for _ in range(self.hidden_layers - 1):  # For additional hidden layers
            self.weights.append(np.random.randn(self.hidden_nodes, self.hidden_nodes + 1))  # Additional hidden layers
        
        self.weights.append(np.random.randn(self.output_nodes, self.hidden_nodes + 1))  # Last hidden layer to output layer

        self.previous_deltas = [np.zeros(weight.shape) for weight in self.weights]



    def activation_func(self, value):
        return 1 / (1 + np.exp(-value))
    
    def activation_func_derivative(self,value):
        activation_function = self.activation_func(value)
        return activation_function * (1 - activation_function)
    
    def predict(self, X):
        X = np.insert(X,0,1,axis=1)  # Add bias to input
        activations = [X.T]  # List to store activations of each layer

        # Forward propagate input through each layer
        for i in range(len(self.weights)):
            h = self.weights[i].dot(activations[-1])  # Compute weighted sum
            # Apply activation function (except for output layer)
            if i < len(self.weights) - 1:
                activation = self.activation_func(h)
                activation = np.insert(activation, 0, 1, axis=0)  # Add bias
            else:
                activation = self.activation_func(h)  # Output layer
            activations.append(activation)

        return activations[-1].T  # Return output of the last layer (predictions)

    
    def forward_propagation(self, X):
        # Initialize lists to store outputs and inputs for each layer
        h_outputs = []
        V_inputs = []

        # Compute hidden layer outputs (h) and inputs (V) for each hidden layer
        for i in range(len(self.weights) - 1):
            if i == 0:
                h = self.weights[i].dot(X.T)
            else:
                h = self.weights[i].dot(V_inputs[-1])
            V = self.activation_func(h)
            V_with_bias = np.insert(V, 0, 1, axis=0)  # Add bias term to the input
            h_outputs.append(h)
            V_inputs.append(V_with_bias)

        # Compute output layer outputs (h2) and inputs (O)
        h2 = self.weights[-1].dot(V_inputs[-1])
        O = self.activation_func(h2)

        return h_outputs, V_inputs, h2, O
    
    def backward_propagation(self, h1_outputs, V1_inputs, h2_output, O_predicted):
        # Update output layer weights
        output_errors = self.Y.T - O_predicted  # Compute output errors
        dO = output_errors * self.activation_func_derivative(h2_output)  # Compute derivative of activation function
        dW_output = self.learning_rate * dO.dot(V1_inputs[-1].T)  # Compute weight gradients for output layer
        # Update output layer weights
        self.weights[-1] += dW_output

        # Initialize delta for next layer
        delta_next = dO

        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            # Compute delta for current hidden layer
            weights_without_bias = self.weights[i + 1][:, 1:]  # Exclude bias weights
            # print("Shapes:")
            # print("weights_without_bias:", weights_without_bias.shape)
            # print("delta_next:", delta_next.shape)

            # Compute delta_current for hidden layer
            delta_current = weights_without_bias.T.dot(delta_next) * self.activation_func_derivative(h1_outputs[i])
            # print("delta_current:", delta_current.shape)
            # print("V1_inputs[i]:", V1_inputs[i].shape)

            # Compute weight gradients for current hidden layer
            # Compute weight gradients for current hidden layer
            # Compute weight gradients for current hidden layer
            dW_hidden = self.learning_rate * delta_current.dot(V1_inputs[i].T)
            # Pad dW_hidden with zeros to match the shape of self.weights[i]
            num_cols_diff = self.weights[i].shape[1] - dW_hidden.shape[1]
            dW_hidden = np.concatenate((dW_hidden, np.zeros((dW_hidden.shape[0], num_cols_diff))), axis=1)

            # print("dW_hidden:", dW_hidden.shape)
            # print("self.weights[i]:", self.weights[i].shape)

            # Update weights for current hidden layer
            self.weights[i] += dW_hidden

            # Update delta for next layer
            delta_next = delta_current


    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            h1_outputs, V1_inputs, h2_output, O_predicted = self.forward_propagation(self.X)

            if self.is_converged(O_predicted):
                break

            # if epoch % 1000 == 0:
                # print(f"{epoch=} ; output={O_predicted} ; error={self.compute_error(O_predicted)}")

            self.backward_propagation(h1_outputs, V1_inputs, h2_output, O_predicted)

        return O_predicted, epoch, self.is_converged(O_predicted)

    def get_scaled_outputs(self):
        return
    
    def compute_error(self, O):
        p = self.X.shape[0]
        output_errors = self.Y.T - O  # (output_nodes, N) - (output_nodes, N) = (output_nodes, N)
        return np.power(output_errors, 2).sum() / p
    
    def compute_deltas(self, indexes: [int]):
        return
    
    def is_converged(self, O):
        expected_outputs_amplitude = 1 - 0  # amplitude of the expected output values (scaled to logistic function range)
        percentage_threshold = 0.1
        return self.compute_error(O) < percentage_threshold * expected_outputs_amplitude

    def __str__(self) -> str:
        output = "---MULTI-LAYER PERCEPTRON---\n"

        _, _, _, O = self.forward_propagation(self.X)

        output += f"Training Error: {self.compute_error(O)}\n"
        output += f"Converged: {self.is_converged(O)}\n"

        return output