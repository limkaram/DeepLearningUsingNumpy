'''
DEEP LEARNING (Spring, 2021)
Implementation of cifar-10 data classification using multi-perceptron (MLP)
'''

import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def sigmoid(x, backward=False):
    if backward:
        dx = (np.exp(-x))/((np.exp(-x)+1)**2)
        return dx
    return 1 / (1 + np.exp(-x))


def softmax(x):
    output = np.zeros_like(x)
    exp = np.exp(x)
    sum_exp = np.sum(np.exp(x), axis=1)
    for i in range(x.shape[0]):
        output[i, :] = exp[i, :] / sum_exp[i]

    return output

def predict(x):
    predict_labels = np.argmax(x, axis=1)

    return predict_labels


def count_correct(predict, label):
    correct = np.sum(label == predict)

    return correct


def feedforward(x, w1, w2, w3):
    y1 = np.matmul(x, w1)
    a1 = sigmoid(y1)
    y2 = np.matmul(a1, w2)
    a2 = sigmoid(y2)
    y3 = np.matmul(a2, w3)

    return a1, a2, y3


def cross_entropy(x, y):
    batch_size = x.shape[0]
    return -np.sum(y * np.log(x + 1e-7)) / batch_size



def Gradient(forward_output, dout, w):
    dw = np.dot(forward_output.T, dout)
    dx = np.dot(dout, w.T)

    return dx, dw

def one_hot_encoding(idx, size):
    # new personal function
    encoded_array = np.zeros((size, ))
    encoded_array[idx] = 1
    return encoded_array

def std_normalization(X):
    # new personal function
    return (X - np.mean(X)) / np.std(X)


'''Main'''
# Load Data
print('Loading the cifar-10 dataset...')
dataset = unpickle('./train_data')
X = dataset[b'data']     # numpy array type
y = dataset[b'labels']     # list type
data_size = len(y)
train_data_size = int(data_size*0.9)
X_train = X[:train_data_size,:]
y_train = y[:train_data_size]
X_val = X[train_data_size:,:]
y_val = y[train_data_size:]
test_dataset = unpickle('./test_data')
X_test = test_dataset[b'data'][:1000,:]
y_test = test_dataset[b'labels'][:1000]
num_class = len(np.unique(y_train))

# one-hot enoding lables data
y_train = np.array([one_hot_encoding(i, num_class) for i in y_train])
y_val = np.array([one_hot_encoding(i, num_class) for i in y_val])
y_test = np.array([one_hot_encoding(i, num_class) for i in y_test])

print('Successfully loaded!')
print('******** data information ********')
print('train data shape:', X_train.shape)
print('train labels shape:', len(y_train))
print('validation data shape:', X_val.shape)
print('validation labels shape:', len(y_val))
print('test data shape:', X_test.shape)
print('test labels shape:', len(y_test))
print('the number of classes:', num_class)
print('**********************************')


# Construct network
# 1) Initialization
# 1-1) hyper-parameters setting
learning_rate = 0.005
epochs = 50
num_node_1 = 512
num_node_2 = 256
batch_size = 64
lambda_ = 10 ** -2


# 1-2) weights initialization
weights_1 = np.random.normal(0, 0.01, (X_train.shape[1], num_node_1))
weights_2 = np.random.normal(0, 0.01, (num_node_1, num_node_2))
weights_3 = np.random.normal(0, 0.01, (num_node_2, num_class))


# 2) Learning
losses_tr = []
losses_val = []
accs_tr = []
accs_val = []

for epoch in range(epochs):
    print('Epoch: ', epoch)
    correct_tr = 0
    loss_tr = 0

    correct_val = 0
    loss_val = 0

    # Training
    for i in range(int(len(X_train) / batch_size)):
        X_batch = std_normalization(X_train[batch_size * i:batch_size * (i + 1)])
        y_batch = y_train[batch_size * i:batch_size * (i + 1)]
        a1, a2, y3 = feedforward(X_batch, weights_1, weights_2, weights_3)
        a3 = softmax(y3)
        loss = cross_entropy(a3, y_batch)
        pred_labels = np.array([one_hot_encoding(i, 10) for i in predict(y3)])
        loss_tr += loss
        correct_tr += count_correct(pred_labels, y_batch)


        # GetGradient
        dxw3 = a3 - y_batch / batch_size
        dx3, dw3 = Gradient(a2, dxw3, weights_3)
        # dw3 = np.dot(a2.T, dxw3)
        # dx3 = np.dot(dxw3, weights_3.T)

        dxw2 = sigmoid(dx3, backward=True) / batch_size
        dx2, dw2 = Gradient(a1, dxw2, weights_2)
        # dw2 = np.dot(a1.T, dxw2)
        # dx2 = np.dot(dxw2, weights_2.T)

        dxw1 = sigmoid(dx2, backward=True) / batch_size
        dx1, dw1 = Gradient(X_batch, dxw1, weights_1)
        # dw1 = np.dot(X_batch.T, dxw1)
        # dx1 = np.dot(dxw1, weights_1.T)

        # Weights update
        weights_1 -= learning_rate * dw1
        weights_2 -= learning_rate * dw2
        weights_3 -= learning_rate * dw3


    # Validation
    for i in range(int(len(X_val) / batch_size)):
        X_ = X_val[batch_size * i:batch_size * (i + 1)]
        X_ = (X_ - np.mean(X_train)) / np.std(X_train)
        y_ = y_val[batch_size * i:batch_size * (i + 1)]
        a1, a2, y3 = feedforward(X_, weights_1, weights_2, weights_3)
        a3 = softmax(y3)
        loss = cross_entropy(a3, y_)
        pred_labels = np.array([one_hot_encoding(i, 10) for i in predict(y3)])
        loss_val += loss
        correct_val += count_correct(pred_labels, y_)


    # Get training accuracy
    accuracy_tr = correct_tr/len(y_train) * 100
    loss_tr = loss_tr/len(y_train)
    print("Train_loss: %.02f" % loss_tr)
    print("Train_acc: %.02f %%" % accuracy_tr)

    # Get validation accuracy
    accuracy_val = correct_val/len(y_val) * 100
    loss_val = loss_val/len(y_val)
    print("Validation_loss: %.02f" % loss_val)
    print("Validation_acc: %.02f %%" % accuracy_val)

    losses_tr.append(loss_tr)
    losses_val.append(loss_val)
    accs_tr.append(accuracy_tr)
    accs_val.append(accuracy_val)


# Show results

# Plot loss and accuracy curve
plt.figure()
plt.plot(losses_tr, 'y', label='Train_loss')
plt.plot(losses_val, 'r', label='Validation_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Train_loss','Validation_loss'])
plt.show()

plt.figure()
plt.plot(accs_tr, 'y', label='Train_accuracy')
plt.plot(accs_val, 'r', label='Validation_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train_acc','Validation_acc'])
plt.show()

# Test
# Calculate test accuracy (final accuracy)
loss_test = 0
correct_test = 0

for i in range(int(len(X_test) / batch_size)):
    X_ = X_test[batch_size * i:batch_size * (i + 1)]
    X_ = (X_ - np.mean(X_train)) / np.std(X_train)
    y_ = y_test[batch_size * i:batch_size * (i + 1)]
    a1, a2, y3 = feedforward(X_, weights_1, weights_2, weights_3)
    a3 = softmax(y3)
    loss = cross_entropy(a3, y_)
    pred_labels = np.array([one_hot_encoding(i, 10) for i in predict(y3)])
    loss_test += loss
    correct_test += count_correct(pred_labels, y_)

accuracy_test = correct_test / len(y_test) * 100
loss_test = loss_test / len(y_test)
print("Test accuracy: %.02f %%" % accuracy_test)

# Nearest Neighbor
