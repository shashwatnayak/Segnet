import keras
import numpy as np
from keras.models import Sequential # for sequential cnn
from keras.layers import Dense,Dropout,Flatten # reg - dropout, flatten for FC
from keras.layers import Conv2D, MaxPooling2D # Inside NN config uses conv 2d and maxpool2d
from keras.optimizers import SGD # sgd used. will consider adam

#for random dummy data
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)


#input is 100x100 with 3 channels

model = Sequential()
# 32 convolutions with 3x3 filters
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) #reg

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)


# Using TensorFlow backend.
# Epoch 1/10
# 2019-01-03 23:28:39.917557: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
# 100/100 [==============================] - 6s 64ms/step - loss: 2.3081
# Epoch 2/10
# 100/100 [==============================] - 6s 57ms/step - loss: 2.3382
# Epoch 3/10
# 100/100 [==============================] - 6s 58ms/step - loss: 2.2872
# Epoch 4/10
# 100/100 [==============================] - 6s 55ms/step - loss: 2.3054
# Epoch 5/10
# 100/100 [==============================] - 6s 56ms/step - loss: 2.2914
# Epoch 6/10
# 100/100 [==============================] - 6s 55ms/step - loss: 2.2853
# Epoch 7/10
# 100/100 [==============================] - 6s 55ms/step - loss: 2.2765
# Epoch 8/10
# 100/100 [==============================] - 6s 55ms/step - loss: 2.2723
# Epoch 9/10
# 100/100 [==============================] - 6s 55ms/step - loss: 2.2858
# Epoch 10/10
# 100/100 [==============================] - 6s 55ms/step - loss: 2.2847
# 20/20 [==============================] - 0s 21ms/step

#Conclusion not that good.
#Will look towards improving this net.
