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

# def count_correct():
#
#
#     return


def feedforward(network, X_batch):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']

    a1 = np.dot(X_batch, W1)
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2)
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)

    return a3


def cross_entropy(x, one_hot_labels):
    batch_size = x.shape[0]
    return -np.sum(one_hot_labels * np.log(x + 1e-7)) / batch_size

# def Gradient():
#
#
#     return


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

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test