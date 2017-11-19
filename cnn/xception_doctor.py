from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
#from xception import Xception

# parameters
import argparse
parser = argparse.ArgumentParser(description='xception doctor')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size')
parser.add_argument('--epoch', type=int, default=10,
                    help='epoch')
parser.add_argument('--rescale', type=int, default=128,
                    help='rescale image size')
parser.add_argument('--part', type=str, default='brain',
                    help='body part')
args = parser.parse_args()

print('####',args.part)

### setup dataset
data_dir = '../data/' + args.part 
target_size = (args.rescale, args.rescale)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    data_dir + '/train',
    target_size=target_size,
    batch_size=args.batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    data_dir + '/validation',
    target_size=target_size,
    batch_size=args.batch_size,
    class_mode='categorical')


### build model
n_labels = train_generator.num_class
print('{} labels'.format(n_labels), train_generator.class_indices)

model = Xception(
    weights=None,
    classes=n_labels)

model.summary()

model.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])


### train
history = model.fit_generator(
    train_generator,
    steps_per_epoch=200,#train_generator.n//batch_size,
    epochs=args.epoch,
    validation_data=validation_generator,
    validation_steps=20)


### save model weights 
import os
result_dir = './models/'
model.save(os.path.join(result_dir, 
    'xception.rescale={}.{}.model.h5'.format(args.rescale, args.part)))
model.save_weights(os.path.join(result_dir, 
    'xception.rescale={}.{}.weghit.h5'.format(args.rescale, args.part)))
