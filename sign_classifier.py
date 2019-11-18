import tensorflow.keras as kr
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import to_categorical

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

(x_train, y_train), (x_test, y_test) = kr.datasets.cifar10.load_data()
 
num_classes = 10
y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test,num_classes)

weight_decay = 1e-4
model = tf.keras.Sequential()

model.add(kr.layers.Conv2D(32, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(kr.layers.Activation('relu'))
model.add(kr.layers.BatchNormalization())
model.add(kr.layers.Conv2D(32, (3,3), padding='same'))
model.add(kr.layers.Activation('relu'))
model.add(kr.layers.BatchNormalization())
model.add(kr.layers.MaxPooling2D(pool_size=(2,2)))
model.add(kr.layers.Dropout(0.2))
 
model.add(kr.layers.Conv2D(64, (3,3), padding='same'))
model.add(kr.layers.Activation('relu'))
model.add(kr.layers.BatchNormalization())
model.add(kr.layers.Conv2D(64, (3,3), padding='same'))
model.add(kr.layers.Activation('relu'))
model.add(kr.layers.BatchNormalization())
model.add(kr.layers.MaxPooling2D(pool_size=(2,2)))
model.add(kr.layers.Dropout(0.3))
 
model.add(kr.layers.Conv2D(128, (3,3), padding='same'))
model.add(kr.layers.Activation('relu'))
model.add(kr.layers.BatchNormalization())
model.add(kr.layers.Conv2D(128, (3,3), padding='same'))
model.add(kr.layers.Activation('relu'))
model.add(kr.layers.BatchNormalization())
model.add(kr.layers.MaxPooling2D(pool_size=(2,2)))
model.add(kr.layers.Dropout(0.4))
 
model.add(kr.layers.Flatten())
model.add(kr.layers.Dense(10, activation='softmax'))
model.summary()
 
#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(x_train)
 
#training
batch_size = 64
 
opt_rms = tf.keras.optimizers.RMSprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=125,\
                    verbose=1,validation_data=(x_test,y_test),callbacks=[kr.callbacks.LearningRateScheduler(lr_schedule)])