import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from network import Model,LayerConv2d,LayerDense,MaxPooling,binarize
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = np.asarray(
    keras.datasets.mnist.load_data())

#Normalization
X_train = (X_train / 255).astype(np.float32)
X_test = (X_test / 255).astype(np.float32)
y_train=binarize(y_train)
y_test=y_test
y_train.shape

model = Model()
model.add(LayerConv2d(1, 3, kernel_size=4, padding=0,
          stride=1, activation_function="relu"))
model.add(LayerConv2d(3, 3, kernel_size=3, padding=0,
          stride=2, activation_function="relu"))
model.add(LayerDense(432, 64, "relu"))
model.add(LayerDense(64, 10, "softmax"))

model.train(X_train,y_train,learning_rate=0.0025,epochs=30,batch_size=2048)
model.save("weights.json")
#model.load("weights.json")

print(model.accuracy(X_test,y_test))

'''
model.forward(X_test[10].reshape(1,28,28,1))
model.backward(X_test[10].reshape(1,28,28,1),binarize(y_test[0].reshape(1,1)).reshape(10,1))
model.layers[0].output[0][18][18]
model.layers[1].output[0][18][18]
a=model.layers[1].gradient.numpy()[0][18][18]
b=model.layers[2].gradient.numpy()[0][18][18]
np.allclose(a,b)


print(np.argmax(model.forward(X_test[10].reshape(1,28,28,1))))
a=model.layers[0].kernels
b=model.layers[0].output
plt.imshow(X_test[10],cmap="binary")
plt.imshow(b[0,:,:,2])
'''