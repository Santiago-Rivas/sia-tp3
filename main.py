import sys
import csv

x_tuple = (float, float, float)
float_array = [float, float, float]


class SimplePerceptron:
    def __init__(self, weights: float_array = [-1, 1, 1], learning_rate: float = 0.01, limit: float = 100):
        self.learning_rate = learning_rate
        self.limit = limit
        self.weights = weights
        self.min_error = sys.maxsize

    def projection(self, x: x_tuple) -> float:
        return x[0] * self.weights[0] + \
            x[1] * self.weights[1] + self.weights[2]

    def step_activation(self, val: float) -> int:
        if (val >= 0):
            return 1
        else:
            return -1

    def predict(self, x: x_tuple) -> int:
        return self.step_activation(self.projection(x))

    def compute_error(self, data: [x_tuple]):
        correct = 0
        total = 0
        for x in data:
            activation = self.step_activation(self.projection(x))
            if activation == x[2]:
                correct += 1
            total += 1
        return 1 - (correct/total)

    def train(self, data: [x_tuple]):
        i = 0
        error = 0
        w_min = 0
        while (self.min_error > 0 and i < self.limit):
            for x in data:
                exitement = self.projection(x)
                activation = self.step_activation(exitement)
                delta_w = tuple(self.learning_rate *
                                (x[2] - activation) * mult for mult in x)
                # self.weights = self.weights + delta_w
                self.weights = tuple(
                    x + y for x, y in zip(self.weights, delta_w))
                error = self.compute_error(data)
                if error < self.min_error:
                    self.min_error = error
                    w_min = self.weights
                i += 1
        self.weights = w_min


if __name__ == "__main__":
    # Initialize empty list to store data
    data = []

    # Read data from CSV file
    with open('train_Y.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            x1, x2, y = map(int, row)
            data.append((x1, x2, y))

    print("AND perceptron")
    simple_perceptron = SimplePerceptron(weights=[1, 1, -1])
    for x in data:
        prediction = simple_perceptron.predict(x)
        print("Vales: (", x[0], ", ", x[1], ")\t",
              "Expexted:", x[2], "\tPrediction:", prediction)

    data = []

    # Read data from CSV file
    with open('train_OR.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            x1, x2, y = map(int, row)
            data.append((x1, x2, y))

    print("OR perceptron")
    simple_perceptron = SimplePerceptron()
    simple_perceptron.train(data)
    for x in data:
        prediction = simple_perceptron.predict(x)
        print("Vales: (", x[0], ", ", x[1], ")\t",
              "Expexted:", x[2], "\tPrediction:", prediction)
