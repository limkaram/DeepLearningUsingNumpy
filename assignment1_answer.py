'''
Medical Artificial Intelligence (Spring, 2021)
Implementation of cifar-10 data classification using multi-perceptron (MLP)
'''

import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def sigmoid(X):
    a = 1 / (1+np.exp(-X))

    return a

def softmax(Z):
    output = np.zeros_like(Z)
    nominator = np.exp(Z)
    denominator = np.sum(np.exp(Z), axis=1)
    for i in range(Z.shape[0]):
        output[i,:] = nominator[i,:]/denominator[i]

    return output

def predict(output):
    predict_labels = np.argmax(output, axis=1)

    return predict_labels

def count_correct(label, predict):
    correct = np.sum(label == predict)

    return correct


"*********************** Edit ***************************"
def feedforward(X, W1, W2, W3, W4, W5):
    Z1 = np.matmul(X, W1)
    a1 = sigmoid(Z1)
    Z2 = np.matmul(a1, W2)
    a2 = sigmoid(Z2)
    Z3 = np.matmul(a2, W3)
    a3 = sigmoid(Z3)
    Z4 = np.matmul(a3, W4)
    a4 = sigmoid(Z4)

    Z = np.matmul(a4, W5)

    return a1, a2, a3, a4, Z

def cross_entropy(label, output, w1, w2, w3, w4, w5, lambda_):
    loss = -np.sum([np.log(output[i, target]) for i, target in enumerate(label)])
    regularization = lambda_/2*(np.sum(w1**2)+np.sum(w2**2)+np.sum(w3**2)+np.sum(w4**2)+np.sum(w5**2))

    return loss + regularization

# function to compute gradients using back-propagation
def Gradient(output, data, w1, w2, w3, w4, w5, a1, a2, a3, a4, label, lambda_):
    one_hot = np.zeros((data.shape[0], num_class))
    for i in range(data.shape[0]):
        one_hot[i, label[i]] = 1

    dw5 = np.matmul(a4.T, (output-one_hot)) + lambda_ * w5
    dw4 = np.matmul(a3.T, np.matmul(output-one_hot, w5.T)*a4*(1-a4)) + lambda_ * w4
    dw3 = np.matmul(a2.T, np.matmul(np.matmul(output-one_hot, w5.T)*a4*(1-a4), w4.T)*a3*(1-a3)) + lambda_ * w3
    dw2 = np.matmul(a1.T, np.matmul(np.matmul(np.matmul(output-one_hot, w5.T)*a4*(1-a4), w4.T)*a3*(1-a3), w3.T) * a2 * (1 - a2)) + lambda_ * w2
    dw1 = np.matmul(data.T, np.matmul(np.matmul(np.matmul(np.matmul(output-one_hot, w5.T)*a4*(1-a4), w4.T)*a3*(1-a3), w3.T) * a2 * (1 - a2), w2.T) * a1 * (1 - a1)) + lambda_ * w1
    return dw1, dw2, dw3, dw4, dw5
"********************************************************"




'''Main'''
# Load Data
# np.random.seed('')
print('Loading the cifar-10 dataset...')
dataset = unpickle('./data/cifar-10/train_data')
X = dataset[b'data']     # numpy array type
y = dataset[b'labels']     # list type
data_size = len(y)
train_data_size = int(data_size*0.9)
X_train = X[:train_data_size,:]
y_train = y[:train_data_size]
X_val = X[train_data_size:,:]
y_val = y[train_data_size:]
test_dataset = unpickle('./data/cifar-10/test_data')
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
lr = 0.005
epoch = 50
num_node_1 = 2048
num_node_2 = 1024
num_node_3 = 512
num_node_4 = 128
batch_size = 64
lambda_ = 10**-2


"************** Edit ********************"
# 1-2) weights initialization
weights_1 = np.random.normal(0, 0.1, (X_train.shape[1], num_node_1))
weights_2 = np.random.normal(0, 0.1, (num_node_1, num_node_2))
weights_3 = np.random.normal(0, 0.1, (num_node_2, num_node_3))
weights_4 = np.random.normal(0, 0.1, (num_node_3, num_node_4))
weights_5 = np.random.normal(0, 0.1, (num_node_4, num_class))

"****************************************"

# 2) Learning
losses_tr = []
losses_val = []
accs_tr = []
accs_val = []

for epochs in range(epoch):
    print('Epoch: ', epochs)
    correct_tr = 0
    loss_tr = 0

    correct_val = 0
    loss_val = 0

    # Training
    for i in range(int(len(X_train)/batch_size)):
        X_ = X_train[batch_size * i:batch_size * (i + 1)]
        X_ = (X_ - np.mean(X_)) / np.std(X_)
        y_ = y_train[batch_size * i:batch_size * (i + 1)]
        a1, a2, a3, a4, z5 = feedforward(X_, weights_1, weights_2, weights_3, weights_4, weights_5)

        outputs = softmax(z5)

        loss = cross_entropy(y_, outputs, weights_1, weights_2, weights_3, weights_4, weights_5, lambda_)
        prediction = predict(outputs)
        loss_tr += loss
        correct_tr += count_correct(y_, prediction)

        # GetGradient
        dw1, dw2, dw3, dw4, dw5 = Gradient(outputs, X_, weights_1, weights_2, weights_3, weights_4, weights_5, a1, a2, a3, a4, y_, lambda_)

        # Weights update
        weights_1 -= lr * dw1
        weights_2 -= lr * dw2
        weights_3 -= lr * dw3
        weights_4 -= lr * dw4
        weights_5 -= lr * dw5

    # Validation
    for i in range(int(len(X_val) / batch_size)):
        X_ = X_val[batch_size * i:batch_size * (i + 1)]
        X_ = (X_ - np.mean(X_)) / np.std(X_)
        y_ = y_val[batch_size * i:batch_size * (i + 1)]
        a1, a2, a3, a4, z5 = feedforward(X_, weights_1, weights_2, weights_3, weights_4, weights_5)
        outputs = softmax(z5)
        loss = cross_entropy(y_, outputs, weights_1, weights_2, weights_3, weights_4, weights_5, lambda_)
        prediction = predict(outputs)
        loss_val += loss
        correct_val += count_correct(y_, prediction)


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




## Show results

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
    X_ = (X_ - np.mean(X_)) / np.std(X_)
    y_ = y_test[batch_size * i:batch_size * (i + 1)]
    a1, a2, a3, a4, z5 = feedforward(X_, weights_1, weights_2, weights_3, weights_4, weights_5)
    outputs = softmax(z5)
    loss = cross_entropy(y_, outputs, weights_1, weights_2, weights_3, weights_4, weights_5, lambda_)
    prediction = predict(outputs)
    loss_test += loss
    correct_test += count_correct(y_, prediction)

accuracy_test = correct_test / len(y_test) * 100
loss_test = loss_test / len(y_test)
print("Test accuracy: %.02f %%" % accuracy_test)

# Nearest Neighbor
