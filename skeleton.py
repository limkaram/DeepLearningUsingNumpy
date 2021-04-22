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

def sigmoid():


    return

def softmax():


    return

def predict():


    return

def count_correct():


    return


def feedforward():


    return

def cross_entropy():


    return

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



# 1-2) weights initialization



# 2) Learning
losses_tr = []
losses_val = []
accs_tr = []
accs_val = []

for  in range():
    print('Epoch: ', )
    correct_tr = 0
    loss_tr = 0

    correct_val = 0
    loss_val = 0

    # Training
    for i in range():


        = feedforward()
        = softmax()
        loss = cross_entropy()
        = predict()
        loss_tr += loss
        correct_tr += count_correct()

        # GetGradient
        = Gradient()

        # Weights update


    # Validation
    for i in range():
        = feedforward()
        = softmax()
        loss = cross_entropy()
        = predict()
        loss_val += loss
        correct_val += count_correct()


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
for i in range():
    = feedforward()
    = softmax()
    loss = cross_entropy()
    = predict()
    loss_test += loss
    correct_test += count_correct()

accuracy_test = correct_test / len(y_test) * 100
loss_test = loss_test / len(y_test)
print("Test accuracy: %.02f %%" % accuracy_test)

# Nearest Neighbor
