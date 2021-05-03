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


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t, params['W1'], params['W2'], params['W3'], lambda_)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


def softmax(Z):
    output = np.zeros_like(Z)
    nominator = np.exp(Z)
    denominator = np.sum(np.exp(Z), axis=1)
    for i in range(Z.shape[0]):
        output[i, :] = nominator[i, :] / denominator[i]

    return output


def predict(output):
    predict_labels = np.argmax(output, axis=1)

    return predict_labels


def count_correct(label, predict):
    correct = np.sum(label == predict)

    return correct

def cross_entropy(y, t, w1, w2, w3, lambda_):
    batch_size = y.shape[0]
    loss = -np.sum([np.log(y[i, target]) for i, target in enumerate(t)])
    regularization = lambda_/2*(np.sum(w1**2)+np.sum(w2**2)+np.sum(w3**2))

    return loss + regularization

"*********************** Edit ***************************"


def feedforward(x):
    conv1 = Convolution(params['W1'], params['b1'], conv_param['stride'], conv_param['pad'])
    relu1 = Relu()
    pool1 = Pooling(pool_h=2, pool_w=2, stride=2)
    affine1 = Affine(params['W2'], params['b2'])
    relu2 = Relu()
    affine2 = Affine(params['W3'], params['b3'])
    last_layer = SoftmaxWithLoss()

    conv1_output = conv1.forward(x)
    relu1_output = relu1.forward(conv1_output)
    pool1_output = pool1.forward(relu1_output)
    affine1_output = affine1.forward(pool1_output)
    relu2_output = relu2.forward(affine1_output)
    affine2_output = affine2.forward(relu2_output)

    return affine2_output, conv1, relu1, pool1, affine1, relu2, affine2, last_layer


# function to compute gradients using back-propagation


def dim2to4(x):
    result = np.zeros((len(x), 3, 32, 32))

    for i in range(x.shape[0]):
        result[i] = x[i].reshape(3, 32, 32)
    return result


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def Gradient(conv1, relu1, pool1, affine1, relu2, affine2, last_layer, lambda_):
    """기울기를 구한다(오차역전파법).

    Parameters
    ----------
    x : 입력 데이터
    t : 정답 레이블

    Returns
    -------
    각 층의 기울기를 담은 사전(dictionary) 변수
        grads['W1']、grads['W2']、... 각 층의 가중치
        grads['b1']、grads['b2']、... 각 층의 편향
    """

    # backward
    dout = 1
    dout = last_layer.backward(dout)

    layers = [conv1, relu1, pool1, affine1, relu2, affine2, last_layer]
    layers.reverse()

    for layer in layers:
        dout = layer.backward(dout)

    # 결과 저장
    grads = {}
    grads['W1'] = conv1.dW + lambda_ * params['W1']
    grads['b1'] = conv1.db + lambda_ * params['b1']
    grads['W2'] = affine1.dW + lambda_ * params['W2']
    grads['b2'] = affine1.db + lambda_ * params['b2']
    grads['W3'] = affine2.dW + lambda_ * params['W3']
    grads['b3'] = affine2.db + lambda_ * params['b3']

    return grads


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 중간 데이터（backward 시 사용）
        self.x = None
        self.col = None
        self.col_W = None

        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


"********************************************************"

'''Main'''
# Load Data
# np.random.seed('')
print('Loading the cifar-10 dataset...')
dataset = unpickle('./train_data')
X = dataset[b'data']  # numpy array type
y = dataset[b'labels']  # list type
data_size = len(y)
train_data_size = int(data_size * 0.9)
X_train = dim2to4(X[:train_data_size, :])
y_train = y[:train_data_size]
X_val = dim2to4(X[train_data_size:, :])
y_val = y[train_data_size:]
test_dataset = unpickle('./test_data')
X_test = dim2to4(test_dataset[b'data'][:1000, :])
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

##

# Construct network
# 1) Initialization

# 1-1) hyper-parameters setting
lr = 0.001
epoch = 50
batch_size = 100
lambda_ = 10 ** -2

conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}
filter_num = conv_param['filter_num']
filter_size = conv_param['filter_size']
filter_pad = conv_param['pad']
filter_stride = conv_param['stride']

hidden_size = 100
output_size = num_class
weight_init_std = 0.01
input_dim = (3, 32, 32)
input_size = input_dim[1]
conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

"************** Edit ********************"
# 1-2) weights initialization
params = {}
params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
params['b1'] = np.zeros(filter_num)
params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
params['b2'] = np.zeros(hidden_size)
params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
params['b3'] = np.zeros(output_size)

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
    for i in range(int(len(X_train) / batch_size)):
        X_ = X_train[batch_size * i:batch_size * (i + 1)]
        X_ = (X_ - np.mean(X_)) / np.std(X_)
        y_ = np.array(y_train[batch_size * i:batch_size * (i + 1)])
        affine2_output, conv1, relu1, pool1, affine1, relu2, affine2, last_layer = feedforward(X_)

        loss = last_layer.forward(affine2_output, y_)
        prediction = predict(affine2_output)

        loss_tr += loss
        correct_tr += count_correct(y_, prediction)

        # GetGradient
        grads = Gradient(conv1, relu1, pool1, affine1, relu2, affine2, last_layer, lambda_)
        W1, b1 = grads['W1'], grads['b1']
        W2, b2 = grads['W2'], grads['b2']
        W3, b3 = grads['W3'], grads['b3']

        # Weights update
        params['W1'] -= lr * W1
        params['W2'] -= lr * W2
        params['W3'] -= lr * W3
        params['b1'] -= lr * b1
        params['b2'] -= lr * b2
        params['b3'] -= lr * b3


    # Validation
    for i in range(int(len(X_val) / batch_size)):
        X_ = X_val[batch_size * i:batch_size * (i + 1)]
        X_ = (X_ - np.mean(X_)) / np.std(X_)
        y_ = np.array(y_val[batch_size * i:batch_size * (i + 1)])
        affine2_output, conv1, relu1, pool1, affine1, relu2, affine2, last_layer = feedforward(X_)
        outputs = softmax(affine2_output)
        loss = last_layer.forward(affine2_output, y_)
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
    y_ = np.array(y_test[batch_size * i:batch_size * (i + 1)])
    affine2_output, conv1, relu1, pool1, affine1, relu2, affine2, last_layer = feedforward(X_)
    outputs = softmax(affine2_output)
    loss = last_layer.forward(affine2_output, y_)
    prediction = predict(outputs)
    loss_test += loss
    correct_test += count_correct(y_, prediction)

accuracy_test = correct_test / len(y_test) * 100
loss_test = loss_test / len(y_test)
print("Test accuracy: %.02f %%" % accuracy_test)

# Nearest Neighbor

