import numpy as np
import src.utils as utils
from src.perceptron import Perceptron
from src.non_linear_perceptron import NonLinearPerceptron
import pandas as pd
import matplotlib.pyplot as plt


def compute_error(perceptron: NonLinearPerceptron, indexes, expected):
    p = len(indexes)
    output_errors = expected - perceptron.get_scaled_outputs_range(indexes)
    return np.power(output_errors, 2).sum() / p


data, expected_values = utils.parse_csv("TP3-ej2-conjunto.csv")

k = 7
print(len(data) / k)
converged = False
l = len(data)
diff = int(l / k)

indexes = []
validation = []

# Define the range
my_range = range(len(data))  # Example range

# Define the number of folds (k)
k = 7

# Calculate the size of each fold
fold_size = len(my_range) // k

# Perform k-fold cross-validation
for i in range(k):
    # Calculate start and end indices for the current fold
    start_index = i * fold_size
    end_index = (i + 1) * fold_size if i < k - 1 else len(my_range)

    # Create training and validation sets
    training_indices = list(
        my_range[:start_index]) + list(my_range[end_index:])
    validation_indices = list(my_range[start_index:end_index])

    indexes.append(training_indices)
    validation.append(validation_indices)
    print(f"Fold {i + 1}:")
    print("Training indices:", training_indices)
    print("Validation indices:", validation_indices)
    print()


b = 1
learning_rate = 0.0001

# Create arrays for perceptrons
non_linear_perceptrons = [NonLinearPerceptron(
    data, expected_values, beta=b, learning_rate=learning_rate)
    for _ in range(3)]

perceptrons_full = [NonLinearPerceptron(
    data, expected_values, beta=b, learning_rate=learning_rate)
    for _ in range(3)]

i = 0
max_epochs = 10000000
epochs = 0
training_epochs = 100
e = 0

df = pd.DataFrame(columns=['k_error_mean', 'k_error_error',
                  'full_error_mean', 'full_error_error'])
df.to_csv("perceptron_errors.csv", mode='w', header=True, index=False)

while not converged:
    # Train perceptrons
    for j in range(3):
        e1, c1 = non_linear_perceptrons[j].train_indexes(
            indexes[i], training_epochs)
        e2, c2 = perceptrons_full[j].train(training_epochs)
        print(e1, c1)
        print(e2, c2)
        print(non_linear_perceptrons[j])
        # print(perceptrons_full[j])
        if e != 0:
            print(df.loc[e - 1])

    # print(epochs)

    epochs += training_epochs
    # Compute errors and update DataFrame
    errors_k = [p.compute_error() for p in non_linear_perceptrons]
    errors_full = [p.compute_error() for p in perceptrons_full]
    df.loc[e] = [
        np.mean(errors_k),      np.std(errors_k, ddof=1) / np.sqrt(3),
        np.mean(errors_full),   np.std(errors_full, ddof=1) / np.sqrt(3)
    ]
    # print(df)

    with open("perceptron_errors.csv", "a") as file:
        file.write(','.join(map(str, df.loc[e])) + '\n')

    i = (i + 1) % k
    e += 1

print(expected_values)
