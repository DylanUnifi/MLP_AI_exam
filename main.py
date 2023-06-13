import random
import matplotlib.pyplot as plt
from keras.metrics import Mean
from load_data import load_mnist
import numpy as np
from pickle import dump, load


# Définir la fonction d'activation ReLU
def relu(x):
    return np.maximum(0, x)


# Définir la fonction d'activation softmax

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


# Définir la classe postive et la classe negative
def to_label(y, neg_class=0, pos_class=1):
    """ """
    n_labels = len(y)
    y_labels = np.zeros(n_labels, dtype=int)

    for i in range(n_labels):
        if y[i] == neg_class:
            y_labels[i] = 0
        elif y[i] == pos_class:
            y_labels[i] = 1
        else:
            print("Error")

    # print(np.count_nonzero(y_labels == 1))
    return y_labels


# Definir un generateur de batch
def iterate_mini_batches(inputs, targets, batch_size, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        end_idx = min(start_idx + batch_size, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt], targets[excerpt]


path = 'data'
# Charger les données d'entraînement
X_train, y_train = load_mnist(path)

# Charger les données de test
X_test, y_test = load_mnist(path, kind='t10k')

# Prétraitement des données
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# Selectioner les classes pour l'apprentissage
selected_classes = [3, 8]
train_index = np.where(np.isin(y_train, selected_classes))[0]
test_index = np.where(np.isin(y_test, selected_classes))[0]
X_train = X_train[train_index]
y_train = y_train[train_index]
X_test = X_test[test_index]
y_test = y_test[test_index]
y_train = to_label(y_train, selected_classes[0], selected_classes[1])
y_test = to_label(y_test, selected_classes[0], selected_classes[1])

# Convertir les étiquettes en "one-hot" encoding
num_classes_train = 2
num_classes_test = 2
y_train_onehot = np.eye(num_classes_train)[y_train]
y_test_onehot = np.eye(num_classes_test)[y_test]

# Définir les dimensions du réseau de neurones
input_size = X_train.shape[1]
hidden_size = 128
output_size = 2

# Initialiser les poids et les biais du réseau de neurones
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Définir le nombre d'itérations, le taux d'apprentissage, la taille des batch et le batch
num_iterations = 10
learning_rate = 0.1
batch_size = 50
batch = []
for i in iterate_mini_batches(X_train, y_train, batch_size, shuffle=True):
    batch.append(i)

# definir les tableaux pour contenir la loss e l'accuracy pour chaque iteration
train_loss_dict = {}
train_accuracy_dict = {}

# Include metrics monitoring
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')

# Boucle d'entraînement
for iteration in range(num_iterations):
    train_loss.reset_state()
    train_accuracy.reset_state()
    print("epoch ={}".format(iteration))
    X_batch, y_batch = batch[random.randint(0, len(batch) - 1)]
    y_batch_onehot = np.eye(num_classes_train)[y_batch]
    # Forward propagation
    z1 = np.dot(X_batch, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)

    # Calcul de la fonction de perte (cross-entropy)
    loss = -np.sum(y_batch_onehot * np.log(a2)) / len(X_batch)
    train_loss(loss)
    # print("lost ={}".format(loss))

    # Calcul dell'accuracy
    y_pred = np.argmax(a2, axis=1)
    accuracy = np.mean(y_pred == y_batch)
    train_accuracy(accuracy)
    # print('Accuracy: %.2f%%' % (accuracy * 100))

    # Backpropagation
    delta2 = (a2 - y_batch_onehot) / len(X_batch)
    delta1 = np.dot(delta2, W2.T) * (a1 > 0)

    # Mise à jour des poids et des biais
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)
    dW1 = np.dot(X_batch.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    train_loss_dict[iteration] = train_loss.result()
    train_accuracy_dict[iteration] = train_accuracy.result()

# Save the training loss values
with open('./train_loss.pkl', 'wb') as file:
    dump(train_loss_dict, file)

# Save the training accuracy values
with open('./train_accuracy.pkl', 'wb') as file:
    dump(train_accuracy_dict, file)

# Load the training loss and accuracy dictionaries
train_loss = load(open('train_loss.pkl', 'rb'))
train_accuracy = load(open('train_accuracy.pkl', 'rb'))

# Retrieve each dictionary's values
train_values = train_loss.values()
accuracy_values = train_accuracy.values()

# Generate a sequence of integers to represent the epoch numbers
epochs = range(num_iterations)

# tracer les courbes de la loss et de l'accuracy en fonction di nombre d'iteration
plt.plot(epochs, train_values, label='Training Loss')
# plt.plot(epochs, accuracy_values, label='Training Accuracy')

plt.ylabel('loss')
plt.xlabel('iteration')

plt.xticks(np.arange(0, num_iterations+1, 1))

plt.show()

# Prétraitement des données de test
X_test = X_test.reshape(X_test.shape[0], -1)

# Prédictions sur les données de test
z1_test = np.dot(X_test, W1) + b1
a1_test = relu(z1_test)
z2_test = np.dot(a1_test, W2) + b2
a2_test = softmax(z2_test)
y_pred = np.argmax(a2_test, axis=1)

# Calcul de l'accuracy
accuracy = np.mean(y_pred == y_test)
print('Accuracy: %.2f%%' % (accuracy * 100))
