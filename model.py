import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from network import Model
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = np.asarray(
    keras.datasets.mnist.load_data())

#Normalization
X_train = (X_train / 255).astype(np.float32)
X_test = (X_test / 255).astype(np.float32)


model = Model()
model.add(LayerConv2d(1, 16, kernel_size=4, padding=2,
          stride=1, activation_function="relu"))
model.add(MaxPooling(3))
model.add(LayerConv2d(16, 8, kernel_size=3, padding=1,
          stride=2, activation_function="relu"))

model.add(MaxPooling(3, stride=2))
model.add(LayerDense(288, 32, "relu"))
model.add(LayerDense(32, 10, "softmax"))


model.train(X_train,y_train,learning_rate=0.02,epochs=10,batch_size=2)
model.save("weights.json")
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
