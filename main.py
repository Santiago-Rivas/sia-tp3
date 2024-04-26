import sys
import csv
import utils
from step_perceptron import stepPerceptron
import numpy as np


# if __name__ == "__main__":
#     # Initialize empty list to store data
#     data = []

#     # Read data from CSV file
#     with open('train_Y.csv', mode='r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip header row
#         for row in reader:
#             x1, x2, y = map(int, row)
#             data.append((x1, x2, y))

#     print("AND perceptron")
#     simple_perceptron = SimplePerceptron(weights=[1, 1, -1])
#     for x in data:
#         prediction = simple_perceptron.predict(x)
#         print("Vales: (", x[0], ", ", x[1], ")\t",
#               "Expexted:", x[2], "\tPrediction:", prediction)

#     data = []

#     # Read data from CSV file
#     with open('train_OR.csv', mode='r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip header row
#         for row in reader:
#             x1, x2, y = map(int, row)
#             data.append((x1, x2, y))

#     print("OR perceptron")
#     simple_perceptron = SimplePerceptron()
#     simple_perceptron.train(data)
#     for x in data:
#         prediction = simple_perceptron.predict(x)
#         print("Vales: (", x[0], ", ", x[1], ")\t",
#               "Expexted:", x[2], "\tPrediction:", prediction)

if __name__ == "__main__":
    data = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    
    expected_outputs = np.apply_along_axis(utils.logical_and, axis=1, arr=data)

    step_perceptron = stepPerceptron(data,expected_outputs,0.01)

    epochs, converged = step_perceptron.train(5)

    if not converged:
        print(f"Did not converge after {epochs} epochs\n")
    else:
        print(f"Finished learning AND at {epochs} epochs")
        print("Output: ", step_perceptron.get_outputs())
        print("Weights: ", step_perceptron.weights)

    print(step_perceptron)
    
    expected_outputs = np.apply_along_axis(utils.logical_xor, axis=1, arr=data)

    step_perceptron = stepPerceptron(data,expected_outputs,0.01)

    epochs, converged = step_perceptron.train(100)

    print("\n----- XOR -----\n")

    if not converged:
        print(f"Did not converge after {epochs} epochs\n")
    else:
        print(f"Finished learning at {epochs} epochs\n")
        print("Output: ", step_perceptron.get_outputs())
        print("Weights: ", step_perceptron.weights)

    print(step_perceptron)

