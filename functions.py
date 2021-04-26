import numpy as np
import pickle

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    result = np.zeros_like(x, dtype=np.float64)

    for i in range(len(x)):
        exp_sum = sum(np.exp(x[i]))
        exp = np.exp(x[i])
        result[i] = exp / exp_sum

    return result


def predict(x):
    predict_labels = np.argmax(x, axis=1)
    return predict_labels

def count_correct(y, y_pred):
    return np.sum(np.argmax(y, axis=1) == y_pred)


def feedforward(network, X_batch):
    W1, W2, W3 = network['w1'], network['w2'], network['w3']

    network['z1'] = np.dot(X_batch, W1)
    network['a1'] = sigmoid(network['z1'])
    network['z2'] = np.dot(network['a1'], W2)
    network['a2'] = sigmoid(network['z2'])
    network['z3'] = np.dot(network['a2'], W3)


def cross_entropy(x, one_hot_labels, w1, w2, w3, lambda_):
    batch_size = x.shape[0]
    loss = -np.sum(one_hot_labels * np.log(x + 1e-7))
    l2_norm = (lambda_ / 2) * (np.sum(w1 ** 2) + np.sum(w2 ** 2) + np.sum(w3 ** 2))

    return loss + l2_norm / batch_size

def Gradient(x, network, error):
    batch_size = x.shape[0]

    w3_grad = np.dot(network['a2'].T, error) / batch_size

    dloss_over_da2 = np.dot(error, network['w3'].T)
    dloss_over_dz2 = dloss_over_da2 * network['a2'] * (1 - network['a2'])
    w2_grad = np.dot(network['a1'].T, dloss_over_dz2) / batch_size

    dloss_over_da1 = np.dot(dloss_over_dz2, network['w2'].T)
    dloss_over_dz1 = dloss_over_da1 * network['a1'] * (1 - network['a1'])
    w1_grad = np.dot(x.T, dloss_over_dz1) / batch_size

    return w1_grad, w2_grad, w3_grad


def one_hot_encoding(x: list, num_class=10):
    result = np.zeros((len(x), num_class))

    for i in range(len(x)):
        label = x[i]
        encoded_array = np.zeros((num_class, ))
        encoded_array[label] = 1
        result[i] = encoded_array

    return result


def batch_normalization(x):
    batch_mean = np.mean(x, axis=0)
    batch_xc = x - batch_mean
    batch_variance = np.mean(batch_xc ** 2, axis=0)
    batch_std = np.sqrt(batch_variance + 10e-7)
    normalized_result = batch_xc / batch_std

    return normalized_result
