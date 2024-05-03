import sys
import csv
import src.utils as utils
from src.step_perceptron import StepPerceptron
from src.linear_perceptron import LinearPerceptron
from src.non_linear_perceptron import NonLinearPerceptron
import numpy as np
import pandas as pd


def parse_csv(path: str):
    with open(path, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        next(csv_reader)

        inputs = []
        expected_outputs = []

        for row in csv_reader:
            inputs.append(row[:-1])
            expected_outputs.append(row[-1])

        # Return as numpy array of float numbers
        return np.array(inputs, dtype=float), np.array(expected_outputs, dtype=float)


if __name__ == "__main__":
    data = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])

    expected_outputs = np.apply_along_axis(utils.logical_and, axis=1, arr=data)

    step_perceptron = StepPerceptron(data, expected_outputs, 0.01)

    epochs, converged = step_perceptron.train(5)

    if not converged:
        print(f"Did not converge after {epochs} epochs\n")
    else:
        print(f"Finished learning AND at {epochs} epochs")
        print("Output: ", step_perceptron.get_outputs())
        print("Weights: ", step_perceptron.weights)

    print(step_perceptron)

    expected_outputs = np.apply_along_axis(utils.logical_xor, axis=1, arr=data)

    step_perceptron = StepPerceptron(data, expected_outputs, 0.01)

    epochs, converged = step_perceptron.train(100)

    print("\n----- XOR -----\n")

    if not converged:
        print(f"Did not converge after {epochs} epochs\n")
    else:
        print(f"Finished learning at {epochs} epochs\n")
        print("Output: ", step_perceptron.get_outputs())
        print("Weights: ", step_perceptron.weights)

    print(step_perceptron)

    print("\n\nLinear Perceptron")

    df = pd.read_csv('TP3-ej2-conjunto.csv')

    # Initialize an empty NumPy array to store the rows
    data = np.empty((0, len(df.columns) - 1),
                    dtype=float)  # Exclude the last column

    expected_values = np.empty((0, len(df.columns) - 1),
                               dtype=float)  # Exclude the last column

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        values = row.iloc[:-1].values
        print("values:", values)
        data = np.append(data, [values], axis=0)
        expected_values = np.append(expected_values, row[-1])

    print(data)
    print(expected_values)

    data, expected_values = parse_csv("TP3-ej2-conjunto.csv")

    # linear_perceptron = LinearPerceptron(data, expected_values)
    # epochs, converged = linear_perceptron.train()

    # print("\n----- LinearPerceptron -----\n")

    # if not converged:
    #     print(f"Did not converge after {epochs} epochs\n")
    # else:
    #     print(f"Finished learning at {epochs} epochs\n")
    #     print("Output: ", linear_perceptron.get_outputs())
    #     print("Weights: ", linear_perceptron.weights)

    # print(linear_perceptron)

    i = 0.5

    non_linear_perceptron = NonLinearPerceptron(
        data, expected_values, beta=i, learning_rate=0.01)
    epochs, converged = non_linear_perceptron.train(10000)

    print("\n----- NonLinearPerceptron -----\n")

    if not converged:
        print(f"Did not converge after {epochs, i} epochs\n")
        print(non_linear_perceptron)
    else:
        print(f"Finished learning at {epochs, i} epochs\n")
        print("Output: ", non_linear_perceptron.get_scaled_outputs())
        print("Weights: ", non_linear_perceptron.weights)
        print(non_linear_perceptron)

    exit()
    for i in np.arange(8.1, 20, 0.1):
        for j in range(0, 1):
            non_linear_perceptron = NonLinearPerceptron(
                 data, expected_values, learning_rate=i, beta=0.06)
            epochs, converged = non_linear_perceptron.train(1000)

            print("\n----- NonLinearPerceptron -----\n")

            if not converged:
                print(f"Did not converge after {epochs, i} epochs\n")
                print(non_linear_perceptron)
            else:
                print(f"Finished learning at {epochs, i} epochs\n")
                print("Output: ", non_linear_perceptron.get_scaled_outputs())
                print("Weights: ", non_linear_perceptron.weights)
                print(non_linear_perceptron)
                exit()
