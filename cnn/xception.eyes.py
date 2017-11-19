from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from xception import Xception



batch_size = 32
epochs = 10
data_dir = '../data/eyes/'
n_labels = 2


model = Xception(
    weights=None,
    classes=n_labels)

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
model.save_weights(os.path.join(result_dir, 'xception.eyes.h5'))
