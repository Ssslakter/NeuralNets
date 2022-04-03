import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from network import Model,LayerConv2d,LayerDense,MaxPooling,binarize
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = np.asarray(
    keras.datasets.mnist.load_data())

#Normalization
X_train = (X_train / 255).astype(np.float64)
X_test = (X_test / 255).astype(np.float64)
y_train=binarize(y_train)
y_test=y_test
y_train.shape

model = Model()
model.add(LayerConv2d(1, 4, kernel_size=4, padding=0,
          stride=1, activation_function="relu"))
model.add(LayerConv2d(4, 3, kernel_size=3, padding=0,
          stride=2, activation_function="relu"))
model.add(LayerDense(432, 32, "relu"))
model.add(LayerDense(32, 10, "softmax"))

#model.train(X_train,y_train,learning_rate=0.002,epochs=30,batch_size=2048)
#model.save("weights.json")
model.load("weights.json")

print(model.accuracy(X_test,y_test))

'''
print(np.argmax(model.forward(X_test[10].reshape(1,28,28,1))))
a=model.layers[0].kernels
b=model.layers[0].output
plt.imshow(X_test[10],cmap="binary")
plt.imshow(b[0,:,:,2])
'''
