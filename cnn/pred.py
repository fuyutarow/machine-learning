import os
import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Add
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

n_labels = 2

model = SqueezeNet(
    weights=None,
    input_shape=(128,128,3),
    classes=n_labels)
#model.load('./models/cheap_x2squeeze.rescale\=128.eyes.model.h5')
model.load_weights('models/x2squeeze.rescale=128.eyes.weghit.h5')


def pred_img(fname):
    img = image.load_img(fname, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds
    #return np.argmax(preds)

#dir_name = '../data/eyes/N/'
dir_name = '../data/eyes/train/AB/'
flist = os.listdir(dir_name)
for f in flist:
    fname = './data'
    a = pred_img(dir_name+f)
    print(a)
    
