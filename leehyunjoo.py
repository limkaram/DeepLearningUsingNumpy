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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    output = np.zeros_like(x)
    expX = np.exp(x)
    sum_expX = np.sum(expX, axis=1)
    for i in range(x.shape[0]):
        output[i,:] = expX[i,:] / sum_expX[i] 
    return output

def predict(output):
    predict_labels = np.argmax(output, axis=1)
    return predict_labels

def count_correct(label, predict):
    correct = np.sum(label == predict)
    return correct


def feedforward(X, W1, W2, W3):
    Z1 = np.matmul(X, W1)
    a1 = sigmoid(Z1)
    Z2 = np.matmul(a1, W2)
    a2 = sigmoid(Z2)
    Z = np.matmul(a2, W3) 
    return a1, a2, Z

def cross_entropy(label, output, w1, w2, w3, lambda_):
    loss = -np.sum(label * np.log(output) + (1-label) * np.log(1-output))
    l2_reg = (lambda_/2)*(np.sum(w1**2)+np.sum(w2**2)+np.sum(w3**2))
    total_loss = loss + l2_reg
    return total_loss

def Gradient():


    return


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
lr = 0.005
epochs = 50
num_node_1 = 512
num_node_2 = 256
batch_size = 64
lambda_ = 10**-2

# 1-2) weights initialization
weight_1 = np.random.normal(0, 0.01, (X_train.shape[1], num_node_1))
weight_2 = np.random.normal(0, 0.01, (num_node_1, num_node_2))
weight_3 = np.random.normal(0, 0.01, (num_node_2, num_class))



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
    for i in range(int(len(X_train)/batch_size)):
        X_ = X_train[batch_size * i:batch_size*(i + 1)]
        X_ = (X_ - np.mean(X_)) / np.std(X_)
        y_ = y_train[batch_size * i:batch_size * (i + 1)]
        a1, a2, z3 = feedforward(X_, weight_1, weight_2, weight_3)
        print(np.argmax(z3, axis=1))
    break
#         output = softmax(z3)
#         loss = cross_entropy(y_, output, weight_1, weight_2, weight_3, lambda_)
#         predict_label = predict(output)
#         loss_tr += loss
#         correct_tr += count_correct(y_, predict_label)
#
#         # GetGradient
#         = Gradient()
#
#         # Weights update
#
#
#     # Validation
#     for i in range(int(len(X_val)/batch_size)):
#         X_ = X_val[batch_size * i:batch_size*(i + 1)]
#         X_ = (X_ - np.mean(X_)) / np.std(X_)
#         y_ = y_val[batch_size * i:batch_size * (i + 1)]
#         a1, a2, z3 = feedforward(X_, weight_1, weight_2, weight_3)
#         output = softmax(z3)
#         loss = cross_entropy(y_, output, weight_1, weight_2, weight_3, lambda_)
#         predict_label = predict(output)
#         loss_val += loss
#         correct_val += count_correct(y_, predict_label)
#
#
#     # Get training accuracy
#     accuracy_tr = correct_tr/len(y_train) * 100
#     loss_tr = loss_tr/len(y_train)
#     print("Train_loss: %.02f" % loss_tr)
#     print("Train_acc: %.02f %%" % accuracy_tr)
#
#     # Get validation accuracy
#     accuracy_val = correct_val/len(y_val) * 100
#     loss_val = loss_val/len(y_val)
#     print("Validation_loss: %.02f" % loss_val)
#     print("Validation_acc: %.02f %%" % accuracy_val)
#
#     losses_tr.append(loss_tr)
#     losses_val.append(loss_val)
#     accs_tr.append(accuracy_tr)
#     accs_val.append(accuracy_val)
#
#
#
# ## Show results
#
# # Plot loss and accuracy curve
# plt.figure()
# plt.plot(losses_tr, 'y', label='Train_loss')
# plt.plot(losses_val, 'r', label='Validation_loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['Train_loss','Validation_loss'])
# plt.show()
#
# plt.figure()
# plt.plot(accs_tr, 'y', label='Train_accuracy')
# plt.plot(accs_val, 'r', label='Validation_accuracy')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend(['Train_acc','Validation_acc'])
# plt.show()
#
# # Test
# # Calculate test accuracy (final accuracy)
# loss_test = 0
# correct_test = 0
# for i in range(int(len(X_test)/batch_size)):
#     X_ = X_test[batch_size * i:batch_size*(i + 1)]
#     X_ = (X_ - np.mean(X_)) / np.std(X_)
#     y_ = y_test[batch_size * i:batch_size * (i + 1)]
#     a1, a2, z3 = feedforward(X_, weight_1, weight_2, weight_3)
#     output = softmax(z3)
#     loss = cross_entropy(y_, output, weight_1, weight_2, weight_3, lambda_)
#     predict_label = predict(output)
#     loss_test += loss
#     correct_test += count_correct(y_, predict_label)
#
# accuracy_test = correct_test / len(y_test) * 100
# loss_test = loss_test / len(y_test)
# print("Test accuracy: %.02f %%" % accuracy_test)
#
# # Nearest Neighbor
