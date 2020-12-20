import numpy as np
import cv2

RO = 0.2  # krok iteracji ρ = 0.2
ALPHA = 0.9  # współczynnik ɑ, przyjmuje się 0.9

# pierwsza warstwa 800 neuronów - wymiary train_data 20x40 (za dużo XD)
# pierwsza warstwa 64 neuronów
# warstwa ukryta - sqrt(pierwsza*ostatnia)
# warstwa ukryta - 47
# ostatnia warstwa 35 neuronów - ilość znaków używanych do tablic

charset = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
           "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
           "N", "O", "P", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")

training_data = []  # tablica zawierajaca dane trenujace z plików graficznych(fontu)
training_labels = np.array([[0 for _ in range(len(charset))] for _ in range(len(charset))])


# ^tablica "labelujaca" dane z training_data. Przy deklaracji wypelniania zerami do dimension = 35x35


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


def _load_training_data():
    """Loads the training data from training_data directory.
    Updates the labels to corresponding chars from charset f.e
    training_labels[1] = [0, 1, 0, 0, 0, ... 0]
    training_labels[2] = [0, 0, 1, 0, 0, ... 0]
    """
    i = 0
    for char in charset:
        image = cv2.imread('rsc/training_data/' + str(char) + '.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(800, 1)
        training_data.append(image / 255)
        training_labels[i][i] = 1
        i += 1


class NeuralNetwork:
    def __init__(self):
        """Ctor"""
        self.neurons = (800, 64, 47, 35)  # 800 to rozmiar obrazu trenujacego
        self.layers = [[np.ndarray, np.ndarray],  # warstwa początkowa [wagi], [bias]
                       [np.ndarray, np.ndarray],  # warstwa ukryta [wagi], [bias]
                       [np.ndarray, np.ndarray]]  # warstwa wyjsciowa [wagi], [bias]
        self.__init_layers()
        self.activations = [0, 0, 0]
        self.derivatives = [0, 0, 0]
        self.delta = [0, 0, 0]

    def __init_layers(self):
        """Initializes all 3 layers with random values (weights and biases)"""
        for layer in range(len(self.layers)):
            self.layers[layer][0] = np.random.rand(self.neurons[layer + 1], self.neurons[layer])  # 800 - wymiary obrazu
            self.layers[layer][1] = np.random.rand(self.neurons[layer + 1], 1)

    def train(self):
        _load_training_data()
        for i in range(len(training_data)):
            self.activations[0] = sigmoid(
                self.layers[0][0] @ training_data[i] + self.layers[0][1])  # warstwa poczatkowa
            for layer in range(1, len(self.layers)):  # propagacja wprzod
                self.activations[layer] = \
                    sigmoid(self.layers[layer][0] @ self.activations[layer - 1] + self.layers[layer][1])

            for layer in range(len(self.layers)):  # propagacja wstecz
                self.derivatives[layer] = sigmoid_derivative(self.activations[layer])

            label = np.array(training_labels[i].reshape(self.neurons[-1], 1))
            self.delta[len(self.layers) - 1] = (label - self.activations[len(self.layers) - 1]) \
                                               * self.derivatives[len(self.layers) - 1]
            for layer in range(len(self.layers) - 2, 0, -1):
                self.delta[layer] = (
                            self.layers[layer + 1][0].transpose() @ self.delta[layer + 1] * self.derivatives[layer])


        # TODO trenowanie
        pass


if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.train()
    print("xd")
