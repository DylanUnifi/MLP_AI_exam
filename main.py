import math
import random
import matplotlib.pyplot as plt
from keras.metrics import Mean
from load_data import load_mnist
import numpy as np
from pickle import dump, load
from sklearn.model_selection import KFold


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


def annotate_axes(axs, text, fontsize=18):
    axs.text(0.5, 0.5, text, transform=axs.transAxes,
             ha="center", va="center", fontsize=fontsize, color="darkgrey")


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

# prepare cross validation
n_fold = 5
kfold = KFold(n_fold, shuffle=True, random_state=1)

fold_test_prediction = {}
# enumerate splits
for train_ix, test_ix in kfold.split(X_train):
    fold = 1
    # select rows for train and test
    trainX, trainY, testX, testY = X_train[train_ix], y_train[train_ix], X_train[test_ix], y_train[test_ix]
    # Convertir les étiquettes en "one-hot" encoding
    num_classes_train = 2
    num_classes_test = 2
    # y_train_onehot = np.eye(num_classes_train)[y_train]
    # y_test_onehot = np.eye(num_classes_test)[y_test]

    # Définir les dimensions du réseau de neurones
    input_size = trainX.shape[1]
    hidden_size = 128
    output_size = 2

    # Initialiser les poids et les biais du réseau de neurones
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    # Définir le nombre d'itérations, le taux d'apprentissage, la taille des batch et le batch
    num_iterations = 1000
    learning_rate = 0.1
    batch_size = 1
    batch = []
    for i in iterate_mini_batches(trainX, trainY, batch_size, shuffle=True):
        batch.append(i)

    # definir les tableaux pour contenir la loss e l'accuracy pour chaque iteration
    train_loss_dict = {}
    train_accuracy_dict = {}
    test_accuracy_dict = {}

    # Include metrics monitoring
    train_loss = Mean(name='train_loss')
    train_accuracy = Mean(name='train_accuracy')
    test_accuracy = Mean(name="test_accuracy")

    # Boucle d'entraînement
    pred_loss = math.inf
    pred_train_acc = -math.inf
    pred_test_acc = -math.inf
    good_iter_loss = []
    good_train_iter_acc = []
    good_test_iter_acc = []
    for iteration in range(num_iterations):
        train_loss.reset_state()
        train_accuracy.reset_state()
        X_batch, y_batch = batch[random.randint(0, len(batch) - 1)]
        y_batch_onehot = np.eye(num_classes_train)[y_batch]
        # Forward propagation
        z1 = np.dot(X_batch, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = softmax(z2)

        # Calcul de la fonction de perte (cross-entropy)
        loss = np.round(-np.sum(y_batch_onehot * np.log(a2)) / len(X_batch), 3)
        if loss < pred_loss:
            pred_loss = loss
            train_loss(loss)
            good_iter_loss.append(iteration)
            train_loss_dict[iteration] = train_loss.result()
            print("New good training lost = {} obtain at epochs = {}".format(loss, iteration))

        # Calcul dell'accuracy sur l'ensemble de train
        y_pred = np.argmax(a2, axis=1)
        accuracy = np.round(np.mean(y_pred == y_batch), 3)
        if accuracy > pred_train_acc:
            pred_train_acc = accuracy
            train_accuracy(accuracy)
            good_train_iter_acc.append(iteration)
            train_accuracy_dict[iteration] = train_accuracy.result()
            print('New good train accuracy = {} obtain at epochs = {}'.format(accuracy * 100, iteration))

        # Prétraitement des données de test
        testX = testX.reshape(testX.shape[0], -1)

        # Prédictions sur les données de test
        z1_test = np.dot(testX, W1) + b1
        a1_test = relu(z1_test)
        z2_test = np.dot(a1_test, W2) + b2
        a2_test = softmax(z2_test)
        y_pred = np.argmax(a2_test, axis=1)

        # Calcul de l'accuracy sur l'ensemble de test
        accuracy = np.round(np.mean(y_pred == testY), 3)
        if accuracy > pred_test_acc:
            pred_test_acc = accuracy
            test_accuracy(accuracy)
            good_test_iter_acc.append(iteration)
            test_accuracy_dict[iteration] = test_accuracy.result()
            print('New test accuracy = {} obtain at epochs = {}'.format(accuracy * 100, iteration))

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

    # Save the training loss values
    with open('./train_loss.pkl', 'wb') as file:
        dump(train_loss_dict, file)

    # Save the training accuracy values
    with open('./train_accuracy.pkl', 'wb') as file:
        dump(train_accuracy_dict, file)

    # Save the test accuracy values
    with open('./test_accuracy.pkl', 'wb') as file:
        dump(test_accuracy_dict, file)

    # Load the training loss and accuracy dictionaries
    train_loss = load(open('train_loss.pkl', 'rb'))
    train_accuracy = load(open('train_accuracy.pkl', 'rb'))
    test_accuracy = load(open('test_accuracy.pkl', 'rb'))

    # Retrieve each dictionary's values
    train_values = train_loss.values()
    train_accuracy_values = train_accuracy.values()
    test_accuracy_values = test_accuracy.values()

    # Generate a sequence of integers to represent the epoch numbers
    # epochs = range(num_iterations)

    # tracer les courbes de la loss et de l'accuracy en fonction di nombre d'iteration
    # plt.plot(epochs, train_values, label='Training Loss')

    # plot Loss
    plt.subplot(311)
    plt.title('Cross Entropy Loss')
    plt.plot(good_iter_loss, train_values, label='Training Loss')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    # plt.yticks(np.arange(0, 1, step=0.10))

    # plot accuracy
    plt.subplot(312)
    plt.title('Classification Accuracy')
    plt.plot(good_test_iter_acc, test_accuracy_values, label='Training Accuracy')
    plt.xlabel('iterations')
    plt.ylabel('Accuracy')
    # plt.yticks(np.arange(0, 1, step=0.10))

    # Prétraitement des données de test
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Prédictions sur les données de test
    z1_test = np.dot(X_test, W1) + b1
    a1_test = relu(z1_test)
    z2_test = np.dot(a1_test, W2) + b2
    a2_test = softmax(z2_test)
    y_pred = np.argmax(a2_test, axis=1)

    index = np.where(np.isin(y_test, 0))
    first_class = index[0].size
    f_good_pred = np.sum(np.isin(y_pred[index[0]], 0) == np.isin(y_test[index[0]], 0))

    index = np.where(np.isin(y_test, 1))
    second_class = index[0].size
    s_good_pred = np.sum(np.isin(y_pred[index[0]], 1) == np.isin(y_test[index[0]], 1))
    fold_test_prediction['fold{}'.format(fold)] = ' '.join((str(first_class), str(second_class), str(f_good_pred),
                                                           str(s_good_pred)))
    break

text = ''
for v in fold_test_prediction.values():
    text = 'First class | Second class | well predict first | well predict second \n' + v
    annotate_axes(plt.axes(313), text, fontsize=10)

plt.show()
