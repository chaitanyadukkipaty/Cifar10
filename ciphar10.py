from keras.datasets import cifar10

(xtrain,ytrain),(xtest,ytest) = cifar10.load_data()

print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)

from keras.utils import to_categorical

ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

print(ytrain.shape,ytest.shape)

xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain = xtrain/255
xtest = xtest/255

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Activation('softmax'))

#Compile model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(xtrain,ytrain,epochs=10,batch_size=200)
#Final evaluation of the model
scores = model.evaluate(xtest,ytest,verbose=0)
print("large CNN Error: ",str(100-scores[1]*100),"%")

model.save("model.h5")
print("model saved as model.h5")

from  keras.models  import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("model saved as model.json file")