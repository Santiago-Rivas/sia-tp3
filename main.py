import sys
import csv
import src.utils as utils
from src.step_perceptron import StepPerceptron
from src.linear_perceptron import LinearPerceptron
import numpy as np
import pandas as pd


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
        data = np.append(data, [values], axis=0)
        expected_values = np.append(expected_values, row[-1])

    print(data)
    print(expected_values)

    linear_perceptron = LinearPerceptron(data, expected_values)
