from src.multi_layer_perceptron import MultiLayerPerceptron

inputs = [[1, 2, 3],
          [1, 2, 4]]

expected = [[6, 3],
            [7, 4]]

hidden_nodes_dimensions = [3, 5]

multi_layer_perceptron = MultiLayerPerceptron(inputs, expected, hidden_nodes_dimensions, len(expected[0]))

multi_layer_perceptron.train()
