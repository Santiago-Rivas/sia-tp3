from src.multi_layer_perceptron import MultiLayerPerceptron

inputs = [[1, 2, 3],
          [1, 2, 4]]

expected = [6,
            7]

hidden_nodes = 3


multi_layer_perceptron = MultiLayerPerceptron(inputs, expected, hidden_nodes, len(expected))


