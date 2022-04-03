import numpy as np
from tensorflow import nn
from tensorflow import raw_ops
from imageProccesing import ImageProccessing
import json
np.random.seed(0)


def binarize(array):
    mask = np.zeros((array.shape[0], 10))
    for i in range(array.shape[0]):
        mask[i][array[i]] = 1
    return mask


def cost(y_hat, y):
    return -np.sum(y * np.log(y_hat))



class Activation:
    def relu(x):
        y = np.copy(x)
        y[y < 0] = 0
        return y

    def sigmoid(x):
        y = np.exp(x) / (1 + np.exp(x))

    def d_sigmoid(x):
        return Activation.sigmoid(x) * (1 - Activation.sigmoid(x))

    def softmax(x):
        y = np.exp(x)
        return y / np.sum(y, axis=0)

    def d_relu(x):
        y = np.copy(x)
        y[y < 0] = 0
        y[y > 0] = 1
        return y


class LayerConv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation_function="relu", kernels=None):
        self.output = None
        self.stride = stride
        self.padding = padding
        if kernels is None:
            self.kernels = self.__initialize_kernel(
                (kernel_size, kernel_size, in_channels, out_channels))
        else:
            self.kernels = kernels
        self.biases = np.zeros((out_channels, 1), dtype=np.float32)

        if activation_function == "relu":
            self.activation = Activation.relu
            self.derivative = Activation.d_relu
        elif activation_function == "sigmoid":
            self.activation = Activation.sigmoid
            self.derivative = Activation.d_sigmoid
        else:
            raise Exception("not supported activation function")

    def __initialize_kernel(self, size):
        stddev = 1 / np.sqrt(np.prod(size))
        return np.random.normal(size=size, scale=stddev).astype(np.float32)

    def forward1(self, image):
        self.linear_comb = ImageProccessing.my_convolution(
            image, self.kernels, self.stride, self.padding) + self.biases.reshape(1, 1, *self.biases.shape[::-1])
        self.output = self.activation(self.linear_comb)

    def forward(self, image):
        image = np.pad(image, ((0, 0), (self.padding, self.padding),
                               (self.padding, self.padding), (0, 0)))
        self.linear_comb = nn.convolution(image, self.kernels, strides=[
            self.stride] * 2, data_format="NHWC").numpy() + self.biases.reshape(1, 1, *self.biases.shape[::-1])
        self.output = self.activation(self.linear_comb)

    def backward(self, delta, input):
        d_b = np.sum(delta, axis=(0, 1, 2))
        conv_padding = [*[0] * 2, *[self.padding]
                        * 2, *[self.padding] * 2, *[0] * 2]
        d_w = raw_ops.Conv2DBackpropFilter(input=input, filter_sizes=self.kernels.shape, out_backprop=delta, strides=[
                                           1, self.stride, self.stride, 1], padding="EXPLICIT", explicit_paddings=conv_padding, data_format='NHWC')
        d_input = raw_ops.Conv2DBackpropInput(input_sizes=input.shape, filter=self.kernels, out_backprop=delta, strides=[
            1, self.stride, self.stride, 1], padding="EXPLICIT", explicit_paddings=conv_padding)
        return d_b, d_w, d_input


class MaxPooling:
    def __init__(self, kernel_size, stride=1):
        self.output = None
        self.stride = stride
        self.kernel_size = kernel_size

    def forward1(self, image):
        self.output = ImageProccessing.my_max_pool(
            image, self.kernel_size, self.padding, self.stride)

    def forward(self, image):
        self.output = nn.max_pool(image, self.kernel_size,
                                  self.stride, "VALID", data_format="NHWC").numpy()

    def backward(self, delta, input):
        d_pooling = raw_ops.MaxPoolGrad(orig_input=input, orig_output=self.output, grad=delta, ksize=[1, self.kernel_size, self.kernel_size, 1],
                                        strides=[1, self.stride, self.stride, 1], padding="VALID")
        return d_pooling


class LayerDense:
    def __init__(self, n_input, n_neurons, activation_function=None, weights=None):
        self.linear_comb = None
        self.output = None
        if activation_function == "relu":
            self.activation = Activation.relu
            self.derivative = Activation.d_relu
        elif activation_function == "sigmoid":
            self.activation = Activation.sigmoid
            self.derivative = Activation.d_sigmoid
        elif activation_function == "softmax":
            self.activation = Activation.softmax
        else:
            raise Exception("not supported activation function")

        if weights is None:
            self.weights = np.random.randn(
                n_neurons, n_input) / np.sqrt(n_neurons)
            self.biases = np.random.randn(n_neurons, 1)
        else:
            self.weights = weights[0]
            self.biases = weights[1]

    def forward(self, input):
        self.linear_comb = np.dot(self.weights, input) + self.biases
        self.output = self.activation(self.linear_comb)

    def backward(self, delta, input_layer):
        d_b = np.sum(delta, axis=1, keepdims=True)
        d_w = np.dot(delta, input_layer.output.T)
        d_input = np.dot(self.weights.T, delta) * input_layer.derivative(
            input_layer.linear_comb)
        return d_b, d_w, d_input


class Model:
    def __init__(self):
        self.size = 0
        self.layers = []

    def add(self, layer):
        self.size += 1
        self.layers.append(layer)

    def forward(self, input):
        self.layers[0].forward(input)
        for i in range(1, self.size):
            if(isinstance(self.layers[i], LayerDense) and isinstance(self.layers[i - 1], (LayerConv2d, MaxPooling))):
                prev_input = self.layers[i - 1].output
                prev_input = prev_input.reshape(
                    prev_input.shape[0], np.prod(prev_input.shape[1:]))
                self.layers[i].forward(prev_input.T)
            else:
                self.layers[i].forward(self.layers[i - 1].output)

        output = self.layers[self.size - 1].output
        return output

    def backward(self, input, y):
        conv_layer_count = 0
        output = self.forward(input)
        m = output.shape[1]

        grads = [0] * self.size
        delta = output - y

        for i in range(self.size - 1, 0, -1):
            if(isinstance(self.layers[i - 1], (LayerConv2d, MaxPooling))):
                prev_input = self.layers[i - 1].output
                # now it will be in default dense layer shape
                prev_input = prev_input.reshape(
                    prev_input.shape[0], np.prod(prev_input.shape[1:])).T

                d_w = np.dot(delta, prev_input.T) / m
                d_b = np.sum(delta, axis=1, keepdims=True) / m

                grads[i] = [d_w, d_b]

                d_wrt_z = np.dot(self.layers[i].weights.T, delta).T
                d_wrt_z = d_wrt_z.reshape(self.layers[i - 1].output.shape)
                delta = self.layers[i -
                                    1].backward(d_wrt_z, self.layers[i - 1].output)
                conv_layer_count = i
                break

            (d_b, d_w, delta) = self.layers[i].backward(
                delta, self.layers[i - 1])
            grads[i] = [d_w / m, d_b / m]

        for i in range(conv_layer_count - 1, 0, -1):
            layer = self.layers[i]
            if(isinstance(layer, LayerConv2d)):
                (d_b, d_w, delta) = layer.backward(
                    delta, self.layers[i - 1].output)
                grads[i] = [d_w, d_b]
            elif(isinstance(layer, MaxPooling)):
                delta = layer.backward(delta, self.layers[i - 1].output)

        layer = self.layers[0]
        (d_b, d_w, delta) = layer.backward(delta, input)
        grads[0] = [d_w, d_b]
        loss = cost(output, y)
        return grads, loss

    def update_weights(self, grads, lr):
        for i in range(self.size):
            layer = self.layers[i]
            if(isinstance(layer, LayerConv2d)):
                layer.kernels -= lr * grads[i][0].numpy()
                layer.biases -= lr * grads[i][1].reshape(layer.biases.shape)
            elif(isinstance(layer, LayerDense)):
                layer.weights -= lr * grads[i][0]
                layer.biases -= lr * grads[i][1]

    def train(self, X, y, learning_rate=0.2, epochs=10, batch_size=400):
        y=binarize(y)
        for i in range(epochs):
            avg_loss = 0
            for j in range(0, X.shape[0], batch_size):
                X_batch = X[j:j + batch_size].reshape(batch_size, 28, 28, 1)
                y_batch = y[j:j + batch_size].reshape(10, batch_size)
                grads, loss = self.backward(X_batch, y_batch)
                self.update_weights(grads, learning_rate)
                avg_loss += loss
            sample_cost = cost(self.forward(X[0].reshape(
                1, 28, 28, 1)), y[0].reshape(10, 1))
            print(f"epoch {i}", "avg loss over batch:", avg_loss / (2*batch_size),"sample loss:",sample_cost)


    def load(self, file_path):
        dict = json.load(open(file_path))
        try:
            for i in range(self.size):
                self.layers[i].weights = np.array(dict[f"layer{i}"]["weights"])
                self.layers[i].biases = np.array(dict[f"layer{i}"]["biases"])
        except:
            raise Exception("network and weights architectures are different")

    def save(self,file_path):
        file = open(file_path, "w")
        data = {}
        for i in range(self.size):
            if(isinstance(self.layers[i],LayerDense)):
                data[f"layer{i}"] = {"weights": self.layers[i].weights.tolist(),
                                     "biases": self.layers[i].biases.tolist()}
            elif(isinstance(self.layers[i],LayerConv2d)):
                data[f"layer{i}"] = {"kernels": self.layers[i].kernels.tolist(),
                                     "biases": self.layers[i].biases.tolist()}

        json.dump(data, file)
        file.close()


    def accuracy(self,X, y):
            acc = 0
            ans = self.forward(X.reshape(X.shape[0],28,28,1))
            predicted = np.argmax(ans,axis=0)
            acc=np.sum(predicted==y)/y.shape[0]
            return acc
