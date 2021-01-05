import numpy as np
import cv2

# pierwsza warstwa 450 neuronów - wymiary train_data 15x30
# pierwsza warstwa 64 neuronów
# warstwa ukryta - sqrt(pierwsza*ostatnia)
# warstwa ukryta - 47
# ostatnia warstwa 35 neuronów - ilość znaków używanych do tablic


charset = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G",
           "H", "I", "J", "K", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")

training_data = []  # tablica zawierajaca dane trenujace z plików graficznych(fontu)
training_labels = np.array([[0] * len(charset) for _ in range(len(charset))])
# ^tablica "labelujaca" dane z training_data. Przy deklaracji wypelniania zerami do dimension = 35x35

RO = 0.5
ALPHA = 0.5

IMG_WIDTH = 15
IMG_HEIGHT = 30
IMG_SIZE = IMG_HEIGHT * IMG_WIDTH


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


def load_training_data():
    """Loads the training data from training_data directory.
        Updates the labels to corresponding chars from charset f.e
        training_labels[1] = [0, 1, 0, 0, 0, ... 0]
        training_labels[2] = [0, 0, 1, 0, 0, ... 0]
        """
    for char in charset:
        image = cv2.imread('rsc/training_data/' + str(char) + '.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # every training data gets resize to 15x30 pixels
        pixels = image.size
        image = image.reshape(pixels, 1)
        training_data.append(image / 255)
    for label in range(len(training_labels)):
        training_labels[label][label] = 1


class NeuralNetworkMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class NeuralNetwork(metaclass=NeuralNetworkMeta):
    def __init__(self):
        """Ctor"""
        self.neurons = (64, 47, 35)
        self.layers = len(self.neurons)
        self.weights = [np.random.rand(self.neurons[0], IMG_SIZE) - 0.5]
        self.biases = [np.random.rand(self.neurons[0], 1) - 0.5]
        self.temp_weights = [np.zeros((self.neurons[0], IMG_SIZE))]
        self.temp_biases = [np.zeros((self.neurons[0], 1))]

        self.activations = []
        self.derivatives = []
        self.delta = []

        self.init_layers()

    def init_layers(self):
        """Initializes all 3 layers with random values (weights and biases)"""
        for layer in range(self.layers - 1):
            self.weights.append(np.random.rand(self.neurons[layer + 1], self.neurons[layer]) - 0.5)
            self.biases.append(np.random.rand(self.neurons[layer + 1], 1) - 0.5)
            self.temp_weights.append(np.zeros((self.neurons[layer + 1], self.neurons[layer])))
            self.temp_biases.append(np.zeros((self.neurons[layer + 1], 1)))

        for layer in range(self.layers):
            self.activations.append(np.zeros((self.neurons[layer], 1)))
            self.derivatives.append(np.zeros((self.neurons[layer], 1)))
            self.delta.append(np.zeros((self.neurons[layer], 1)))

    def train(self, it, prgbar):
        """Trains the network for given iterations
        :param: it: Number of iterations for network to be trained"""
        load_training_data()
        samples = list(range(len(training_data[0])))
        for dab in range(it):
            if dab % (it / 100) == 0:
                prgbar.setValue(dab / it * 100)
            sample = samples[dab % len(training_data)]
            current_label = np.array(training_labels[sample]).reshape((self.neurons[-1], 1))
            self.activations[0] = sigmoid(np.dot(self.weights[0], training_data[sample]) + self.biases[0])
            for layer in range(self.layers - 1):
                self.activations[layer + 1] = sigmoid(np.dot(self.weights[layer + 1], self.activations[layer]) +
                                                      self.biases[layer + 1])
            for layer in range(self.layers):
                self.derivatives[layer] = sigmoid_derivative(self.activations[layer])

            self.delta[self.layers - 1] = (current_label - self.activations[self.layers - 1]) * \
                                          self.derivatives[self.layers - 1]
            for layer in range(self.layers - 2, 0, -1):
                self.delta[layer] = np.dot(self.weights[layer + 1].T, self.delta[layer + 1]) * self.derivatives[layer]

            self.update_weights(sample)

    def update_weights(self, sample):
        """Updates weights and biases"""
        self.temp_weights[0] = ALPHA * self.temp_weights[0] + RO * \
                               (np.tile(self.delta[0], (1, self.weights[0].shape[1]))) * \
                               (np.tile(training_data[sample].T, (self.weights[0].shape[0], 1)))
        self.biases[0] = RO * self.delta[0] + ALPHA * self.temp_biases[0]
        self.weights[0] += self.temp_weights[0]
        self.biases[0] += self.temp_biases[0]

        for layer in range(1, self.layers):
            self.temp_weights[layer] = ALPHA * self.temp_weights[layer] + RO * \
                                       (np.tile(self.delta[layer], (1, self.weights[layer].shape[1]))) * \
                                       (np.tile(self.activations[layer - 1].T, (self.weights[layer].shape[0], 1)))
            self.biases[layer] = RO * self.delta[layer] + ALPHA * self.temp_biases[layer]
            self.weights[layer] += self.temp_weights[layer]
            self.biases[layer] += self.temp_biases[layer]

    def estimate(self, img_char):
        """Estimates based on what network "knows" what character
        is on provided image
        :param: img_char: image read by cv2 function imread
        :return: index: index of character located in the charset list"""
        img_char = cv2.imread(img_char)
        img_char = cv2.cvtColor(img_char, cv2.COLOR_BGR2GRAY)
        img_char = cv2.resize(img_char, (IMG_WIDTH, IMG_HEIGHT))
        img_char = img_char.reshape(IMG_SIZE, 1)
        img_char = img_char / 255

        activations = sigmoid(np.dot(self.weights[0], img_char) + self.biases[0])
        for layer in range(1, self.layers):
            activations = sigmoid(np.dot(self.weights[layer], activations) + self.biases[layer])

        index = int(activations.argmax(axis=0))
        print(str(int(activations.argmax(axis=0))) + ": " + str(float(activations[index]) * 100))
        print(f"Found char: {charset[index]}")
        return charset[index]


if __name__ == '__main__':
    nn = NeuralNetwork()
    registry = ''

    while True:
        charac = input("Podaj litere")
        test = cv2.imread('tmp/' + str(charac) + '.jpg')
        registry += charset[nn.estimate(test)]
        print(registry)
