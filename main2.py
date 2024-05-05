from src.multi_layer_perceptron import MultiLayerPerceptron
import numpy as np

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
	print("\n----- XOR -----\n")

	inputs = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
	expected_outputs = np.array([1, 1, -1, -1])

	multilayer_perceptron = MultiLayerPerceptron(0.1, inputs, 2,10, 1, expected_outputs)

	_, epochs, _ = multilayer_perceptron.train(10000)
	print("\nStopped training at epoch: ", epochs)
	print(multilayer_perceptron)


	print("\n----- EVEN OR ODD DIGITS -----\n")
	inputs, expected_outputs = parse_digits(f"TP3-ej3-digitos.txt")

	is_digit_even = np.vectorize(lambda digit: 1 if digit % 2 == 0 else -1)(expected_outputs)
	multilayer_perceptron = MultiLayerPerceptron(0.1, inputs, 10, 1,1, is_digit_even)

	_, epochs, _ = multilayer_perceptron.train(10000)
	print("\nStopped training at epoch: ", epochs)
	print(multilayer_perceptron)

	print("\n----- GUESS THE DIGITS -----\n")

	# Same input as the previous experiment
	# expected_output for 0 is [1 0 0 .... 0], for 1 is [0 1 0 .... 0], for 9 is [0 0 .... 0 1]
	multilayer_perceptron = MultiLayerPerceptron(1, inputs,10,1, 10, np.identity(10))

	_, epochs, _ = multilayer_perceptron.train(10000)

	print("\nStopped training at epoch: ", epochs)
	print(multilayer_perceptron)

	#inputs, _ = parse_digits(f"{settings.Config.data_path}/{settings.multilayer_perceptron.predicting_digit}_with_noise.txt")

	# output = multilayer_perceptron.predict(np.array([inputs[0]]))
	# print(output)
	# print(f"\nPrediction is {np.argmax(output)}")

	# visualize_digit(inputs[0])

