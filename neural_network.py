import numpy as np
import cv2


def sigmoid(array: np.array) -> np.array:
    """
    1/(1+exp(-S))
    """
    aux_list = []
    for x in array:
        result = 1 / (1 + np.exp(-x))
        aux_list.append(result)
    sigmoid_result = np.array(aux_list)
    return sigmoid_result


def sigmoid_derivative(sig: np.array) -> np.array:
    """
    f'(x) = x*(1-x)
    """
    aux_list = []
    for x in sig:
        result = x * (1 - x)
        aux_list.append(result)
    derivative = np.array(aux_list)
    return derivative


class NeuralNetwork:
    def __init__(self):
        self.training_data = []
        self.charset = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G",
                        "H", "I", "J", "K", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
        self.training_labels = np.array([[0] * len(self.charset) for _ in range(len(self.charset))])

        self.neurons = (64, 47, 35)
        self.layers = len(self.neurons)
        self.weights = [np.random.rand(self.neurons[0], 450) - 0.5]
        self.biases = [np.random.rand(self.neurons[0], 1) - 0.5]
        self.temp_weights = [np.zeros((self.neurons[0], 450))]
        self.temp_biases = [np.zeros((self.neurons[0], 1))]

        self.activations = []
        self.derivatives = []
        self.delta = []
        self.RO = 0.5
        self.ALPHA = 0.5

        self.init_layers()
        self.load_training_data()
        self.train()

    def load_training_data(self):
        for char in range(len(self.charset)):
            image = cv2.imread('rsc/training_data/' + str(char) + '.png')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (15, 30))  # every training data gets resize to 15x30 pixels
            pixels = image.size
            image = image.reshape(pixels, 1)
            self.training_data.append(image/255)
        for label in range(len(self.training_labels)):
            self.training_labels[label][label] = 1

    def init_layers(self):
        for layer in range(self.layers - 1):
            self.weights.append(np.random.rand(self.neurons[layer + 1], self.neurons[layer]) - 0.5)
            self.biases.append(np.random.rand(self.neurons[layer + 1], 1) - 0.5)
            self.temp_weights.append(np.zeros((self.neurons[layer + 1], self.neurons[layer])))
            self.temp_biases.append(np.zeros((self.neurons[layer + 1], 1)))
        for layer in range(self.layers):
            self.activations.append(np.zeros((self.neurons[layer], 1)))
            self.derivatives.append(np.zeros((self.neurons[layer], 1)))
            self.delta.append(np.zeros((self.neurons[layer], 1)))

    def train(self):
        it = 10000
        samples = list(range(len(self.training_data[0])))
        for dab in range(it):
            if dab % (it/100) == 0:
                print(str(dab/it*100))
            sample = samples[dab % len(self.training_data)]
            current_label = np.array(self.training_labels[sample]).reshape((self.neurons[-1], 1))
            self.activations[0] = sigmoid(np.dot(self.weights[0], self.training_data[sample]) + self.biases[0])
            for layer in range(self.layers - 1):
                self.activations[layer + 1] = sigmoid(np.dot(self.weights[layer + 1], self.activations[layer]) +
                                                      self.biases[layer + 1])
            for layer in range(self.layers):
                self.derivatives[layer] = sigmoid_derivative(self.activations[layer])

            self.delta[self.layers - 1] = (current_label - self.activations[self.layers - 1]) * \
                                           self.derivatives[self.layers - 1]
            for layer in range(self.layers - 2, 0, -1):
                self.delta[layer] = np.dot(self.weights[layer + 1].T, self.delta[layer + 1]) * self.derivatives[layer]
            # Updating weights and biases
            self.temp_weights[0] = self.ALPHA * self.temp_weights[0] + self.RO * \
                                   (np.tile(self.delta[0], (1, self.weights[0].shape[1]))) * \
                                   (np.tile(self.training_data[sample].T, (self.weights[0].shape[0], 1)))
            self.biases[0] = self.RO * self.delta[0] + self.ALPHA * self.temp_biases[0]
            self.weights[0] += self.temp_weights[0]
            self.biases[0] += self.temp_biases[0]

            for layer in range(1, self.layers):
                self.temp_weights[layer] = self.ALPHA * self.temp_weights[layer] + self.RO * \
                                           (np.tile(self.delta[layer], (1, self.weights[layer].shape[1]))) * \
                                           (np.tile(self.activations[layer - 1].T, (self.weights[layer].shape[0], 1)))
                self.biases[layer] = self.RO * self.delta[layer] + self.ALPHA * self.temp_biases[layer]
                self.weights[layer] += self.temp_weights[layer]
                self.biases[layer] += self.temp_biases[layer]

    def estimate(self, img_char):
        img_char = cv2.cvtColor(img_char, cv2.COLOR_BGR2GRAY)
        img_char = cv2.resize(img_char, (15, 30))
        img_char = img_char.reshape(450, 1)
        img_char = img_char / 255

        activations = sigmoid(np.dot(self.weights[0], img_char) + self.biases[0])
        for layer in range(1, self.layers):
            activations = sigmoid(np.dot(self.weights[layer], activations) + self.biases[layer])

        index = int(activations.argmax(axis=0))
        print(str(int(activations.argmax(axis=0))) + ": " + str(float(activations[index]) * 100))
        print(f"Found char: {self.charset[index]}")
        return index


if __name__ == '__main__':
    nn = NeuralNetwork()
    test = cv2.imread('xd.jpg')
    nn.estimate(test)
