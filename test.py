import cv2

training_data = []
charset = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G",
                        "H", "I", "J", "K", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
training_labels = [[0]*len(charset) for _ in range(len(charset))]

for char in range(len(charset)):
    image = cv2.imread('rsc/training_data/' + str(char) + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (15, 30))  # every training data gets resize to 15x30 pixels
    pixels = image.size
    image = image.reshape(pixels, 1)
    training_data.append(image/255)
for label in range(len(training_labels)):
    training_labels[label][label] = 1

print(training_data)
print(training_labels)