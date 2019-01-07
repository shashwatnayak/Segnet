import keras
from keras.layers import Conv2D,MaxPooling,Dense
from keras.layers import AveragePooling2D,Flatten,Dropout
from keras.optimizers import Adam,SGD
from keras.models import Sequential




model = Sequential()

model.add(Conv2D(6,(3,3),activation = 'relu',input_shape = (32,32,1)))
model.add(AveragePooling2D())

model.add(Conv2D(16,(3,3),activation = 'relu'))
model.add(AveragePooling2D())

model.add(Flatten())
model.add(Dense(120,activation = 'relu'))
model.add(Dense(84,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))
