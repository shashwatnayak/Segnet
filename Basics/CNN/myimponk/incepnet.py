import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Flatten,Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.models import model_from_json


(X_train,y_train), (X_text,y_text) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255.0
X_test = X_test/255.0

y_train = np_utils.to_cateforically(y_train)
y_test = np_utils.to_cateforically(y_test)

input_img = Input(shape = (32,32,3))


tower_1 = Conv2D(64,(1,1),padding = 'same',activation = 'relu')(input_img)
tower_1 = Conv2D(64,(3,3),padding = 'same',activation = 'relu')(tower_1)

tower_2 = Conv2D(64,(1,1),padding = 'same',activation = 'relu')(input_img)
tower_2 = Conv2D(64,(5,5),padding = 'same',activation = 'relu')(tower_2)

tower_3 = MaxPooling2D((3,3),strides = (1,1),padding = 'same')(input_img)
tower_3 = Conv2D(64,(1,1),padding = 'same',activation = 'relu')(tower_3)

output = keras.layers.concatenate([tower_1,tower_2,tower_3], axis = 3)

output = Flatten()(output)
out = Dense(10,activation = 'softmax')(output)


model = Model(inputs = input_img,output = out)
print model.summary()

