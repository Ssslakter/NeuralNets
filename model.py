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


model = Model()
model.add(LayerConv2d(1, 16, kernel_size=4, padding=2,
          stride=2, activation_function="relu"))
model.add(MaxPooling(3))
model.add(LayerConv2d(16, 10, kernel_size=3, padding=0,
          stride=1, activation_function="relu"))
model.add(MaxPooling(2))
model.add(LayerDense(1000, 64, "relu"))
model.add(LayerDense(64, 10, "softmax"))

model.train(X_train,y_train,learning_rate=0.02,epochs=30,batch_size=3000)
model.save("weights.json")
#model.load("weights.json")
print(model.accuracy(X_test[:1000],y_test[:1000]))


'''
# plt.show()
test = np.loadtxt('five.txt', delimiter=",")
plt.imshow(X_train[0], cmap=plt.cm.binary)
plt.show()
plt.imshow(test, cmap=plt.cm.binary)
plt.show()

model.train(X_train, y_train, 0.002, 1, 1)
model.save()

accuracy(10000, X_train, y_train)
accuracy(10000, X_test, y_test)
'''
