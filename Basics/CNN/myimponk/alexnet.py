from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD


x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)


model = Sequential()

model.add(Conv2D(96,(11,11), input_shape = (224,224,3),padding = 'same',kernel_regularizer=l2(l2_reg)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(5,5),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation='relu')
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(ZeroPadding((1,1)))
model.add(Conv2D(512,(3,3)padding = 'same'))
model.add(BatchNormalization())
model.add(Activation='relu')
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(ZeroPadding((1,1)))
model.add(Conv2D(512,(3,3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(ZeroPadding((1,1)))
model.add(Conv2D(1024,(3,3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(3072))
model.add(BatchNormalization())
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))


