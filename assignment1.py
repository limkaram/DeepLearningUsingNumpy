'''
DEEP LEARNING (Spring, 2021)
Implementation of cifar-10 data classification using multi-perceptron (MLP)
'''
import numpy as np
import matplotlib.pyplot as plt
from functions import *

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
epochs = 100
num_node_1 = 512
num_node_2 = 256
batch_size = 64
lambda_ = 10 ** -2


# 1-2) weights initialization
network = {}
network['w1'] = np.random.normal(0, 0.01, (X_train.shape[1], num_node_1))
network['w2'] = np.random.normal(0, 0.01, (num_node_1, num_node_2))
network['w3'] = np.random.normal(0, 0.01, (num_node_2, num_class))


# 2) Learning
losses_tr = []
losses_val = []
accs_tr = []
accs_val = []

for epoch in range(1, epochs+1):
    print('Epoch: ', epoch)
    correct_tr = 0
    loss_tr = 0

    correct_val = 0
    loss_val = 0

    # Training
    for i in range(int(len(X_train) / batch_size)):
        X_batch = batch_normalization(X_train[batch_size * i:batch_size * (i + 1), :])
        # X_batch = X_train[batch_size * i:batch_size * (i + 1), :]
        y_batch = one_hot_encoding(y_train[batch_size * i:batch_size * (i + 1)], num_class=10)
        # y_batch = y_train[batch_size * i:batch_size * (i + 1)]
        feedforward(network, X_batch)
        network['a3'] = softmax(network['z3'])

        loss = cross_entropy(network['a3'], y_batch, network['w1'], network['w2'], network['w2'], lambda_)
        y_pred = predict(network['z3'])

        loss_tr += loss
        correct_tr += count_correct(y_batch, y_pred)

        # GetGradient
        dloss_over_dz3 = -(y_batch - network['a3'])  # error
        w1_grad, w2_grad, w3_grad = Gradient(X_batch, network, error=dloss_over_dz3)

        # Weights update
        network['w1'] -= learning_rate * w1_grad
        network['w2'] -= learning_rate * w2_grad
        network['w3'] -= learning_rate * w3_grad


    # Validation
    for i in range(int(len(X_val) / batch_size)):
        X_val_batch = batch_normalization(X_val[batch_size * i:batch_size * (i + 1), :])
        # X_batch = X_train[batch_size * i:batch_size * (i + 1), :]
        y_val_batch = one_hot_encoding(y_val[batch_size * i:batch_size * (i + 1)], num_class=10)
        # y_batch = y_train[batch_size * i:batch_size * (i + 1)]

        feedforward(network, X_val_batch)
        network['a3'] = softmax(network['z3'])
        loss = cross_entropy(network['a3'], y_val_batch, network['w1'], network['w2'], network['w3'], lambda_)
        y_pred = predict(network['z3'])
        loss_val += loss
        correct_val += count_correct(y_val_batch, y_pred)


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
    X_test_batch = batch_normalization(X_test[batch_size * i:batch_size * (i + 1), :])
    y_test_batch = one_hot_encoding(y_val[batch_size * i:batch_size * (i + 1)], num_class=10)

    feedforward(network, X_test_batch)
    network['a3'] = softmax(network['z3'])
    loss = cross_entropy(network['a3'], y_test_batch, network['w1'], network['w2'], network['w3'], lambda_)
    y_pred = predict(network['z3'])
    loss_test += loss
    correct_test += count_correct(y_test_batch, y_pred)

accuracy_test = correct_test / len(y_test) * 100
loss_test = loss_test / len(y_test)
print("Test accuracy: %.02f %%" % accuracy_test)

# # Nearest Neighbor
