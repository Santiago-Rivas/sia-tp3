import sys
import csv
from typing import Optional
import numpy as np


x_tuple = (float, float, float)
float_array = [float, float, float]


class Perceptron:
    def __init__(self,  data: np.array, expected_value: np.array, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.min_error = sys.maxsize
        self.data = np.insert(data,0,1,axis=1)
        self.weights = np.zeros(self.data.shape[1])
        self.expected_value = expected_value

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

    # def compute_error(self, data: [x_tuple]):
    #     correct = 0
    #     total = 0
    #     for x in data:
    #         activation = self.step_activation(self.projection(x))
    #         if activation == x[2]:
    #             correct += 1
    #         total += 1
    #     return 1 - (correct/total)

    def train(self,epochs : Optional[int] = 1000):
        # i = 0
        # error = 0
        # w_min = 0
        for epoch in range(epochs):
            self.update_weights()

            #Me fijo si el error esta dentro de los margenes que busco
            if self.is_converged():
                break
        return epoch+1, self.is_converged()
        # while (self.min_error > 0 and i < self.limit):
        #     for x in data:
        #         exitement = self.projection(x)
        #         activation = self.step_activation(exitement)
        #         delta_w = tuple(self.learning_rate *
        #                         (x[2] - activation) * mult for mult in x)
        #         # self.weights = self.weights + delta_w
        #         self.weights = tuple(
        #             x + y for x, y in zip(self.weights, delta_w))
        #         error = self.compute_error(data)
        #         if error < self.min_error:
        #             self.min_error = error
        #             w_min = self.weights
        #         i += 1
        # self.weights = w_min
    def update_weights(self):
        deltas = self.compute_deltas()
        self.weights = self.weights + np.sum(deltas,axis=0)


    def __str__(self) -> str:
        output = "Expected - Actual\n"

        for expected, actual in zip(self.expected_value, self.get_outputs()):
            output += f"{expected:<10} {actual}\n"

        output += f"\nWeights: {self.weights}"

        return output
    

    def get_outputs(self):
        excitations = np.dot(self.data,self.weights)
        return np.vectorize(self.activation_func)(excitations)

    def compute_error(self):
        raise NotImplementedError

    def is_converged(self):
        raise NotImplementedError

    def compute_deltas(self):
        raise NotImplementedError

    def activation_func(self,value):
        raise NotImplementedError