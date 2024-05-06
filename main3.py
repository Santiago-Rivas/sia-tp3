import csv
from src.multi_layer_perceptron import MultiLayerPerceptron
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def parse_digits(path: str):
    with open(path, 'r') as f:
        data = f.read().splitlines()  # read the file and split into lines

    digit_images = []
    
    for i, line in enumerate(data):
        if i % 7 == 0:  # if we're at the start of a new item
            digit_img = []  # create a new digit_img array
        numbers = line.strip().split()  # remove trailing spaces and split into individual numbers
        digit_img.extend(numbers)  # add the numbers to the current digit_img
        if i % 7 == 6:  # if we're at the end of an digit_img
            digit_images.append(digit_img)  # add the digit_img to the list of digit_images

    digits = np.arange(10)

    return np.array(digit_images, dtype=float), digits

if __name__ == "__main__":
    num_runs = 3

    # Initialize lists to store epochs and errors for each run
    all_epochs = []
    all_errors = []

    print("\n----- EVEN OR ODD DIGITS -----\n")
    inputs, expected_outputs = parse_digits(f"TP3-ej3-digitos.txt")

    is_digit_even = np.vectorize(lambda digit: 1 if digit % 2 == 0 else -1)(expected_outputs)

    # Run the configuration multiple times
    for run in range(num_runs):
        print(f"\n--- Run {run + 1} ---")
        multilayer_perceptron = MultiLayerPerceptron(0.1, inputs, 10, 10, 1, is_digit_even)
        _, epochs, _ = multilayer_perceptron.train(1000000)
        print("Stopped training at epoch:", epochs)
        print(multilayer_perceptron)
        
        # Append epochs and errors to lists
        all_epochs.append(epochs)
        all_errors.append(multilayer_perceptron.compute_error(multilayer_perceptron.predict(inputs)))

    # Calculate mean epochs and mean error
    mean_epochs = np.mean(all_epochs)
    mean_error = np.mean(all_errors)

    # Save epochs, errors, mean epochs, and mean error to a CSV file
    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'Epochs', 'Error'])
        for i in range(num_runs):
            writer.writerow([i + 1, all_epochs[i], all_errors[i]])
        writer.writerow(['Mean', mean_epochs, mean_error])

    data = pd.read_csv('results.csv')

# Plot epochs vs. run
    plt.figure(figsize=(10, 6))
    plt.plot(data['Run'], data['Epochs'], marker='o', linestyle='-')
    plt.xlabel('Run')
    plt.ylabel('Epochs')
    plt.title('Epochs vs. Run')
    plt.grid(True)
    plt.show()
