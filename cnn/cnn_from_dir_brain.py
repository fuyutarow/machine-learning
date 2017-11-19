from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


batch_size = 32
epochs = 10
data_dir = '../data/brain/'
n_labels = 4

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(128, 128, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_labels))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    data_dir + 'train',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    data_dir + 'validation',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical')

print(validation_generator.n//batch_size)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=200,#train_generator.n//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=20)

import os
result_dir = './models/'
model.save_weights(os.path.join(result_dir, 'smallcnn.brain.h5'))
