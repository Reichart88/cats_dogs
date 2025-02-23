from keras.applications import MobileNet
conv_base = MobileNet(weights='imagenet',include_top=False,input_shape=(150, 150, 3))

!wget https://storage.yandexcloud.net/academy.ai/cat-and-dog.zip
!unzip -qo "cat-and-dog" -d ./temp


import os
import shutil
from keras import layers
from keras import models
from keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_PATH = './temp/training_set/training_set/'
BASE_DIR = './dataset/'

CLASS_LIST = sorted(os.listdir(IMAGE_PATH))
CLASS_COUNT = len(CLASS_LIST)
if os.path.exists(BASE_DIR):
  shutil.rmtree(BASE_DIR)

os.mkdir(BASE_DIR)

train_dir = os.path.join(BASE_DIR, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(BASE_DIR, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(BASE_DIR, 'test')
os.mkdir(test_dir)

def create_dataset(
    img_path: str,
    new_path: str,
    class_name: str,
    start_index: int,
    end_index: int
):
    src_path = os.path.join(img_path, class_name)
    dst_path = os.path.join(new_path, class_name)
    class_files = os.listdir(src_path)
    os.mkdir(dst_path)
    for fname in class_files[start_index : end_index]:
      src = os.path.join(src_path, fname)
      dst = os.path.join(dst_path, fname)
      shutil.copyfile(src, dst)

for class_label in range(CLASS_COUNT):
  class_name = CLASS_LIST[class_label]
  create_dataset(IMAGE_PATH, train_dir, class_name, 0, 3000)
  create_dataset(IMAGE_PATH, validation_dir, class_name, 3000, 3500)
  create_dataset(IMAGE_PATH, test_dir, class_name, 3500, 4005)

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 40

def extract_features(directory, sample_count):
  features = np.zeros(shape=(sample_count, 4, 4, 1024))
  labels = np.zeros(shape=(sample_count, 2))

  generator = datagen.flow_from_directory(
      directory,
      target_size=(150, 150),
      batch_size=batch_size,
      class_mode='categorical',
  )
  i = 0
  for input_batch, labels_batch in generator:
    features_batch = conv_base.predict(input_batch, verbose = 0)
    features[i * batch_size : (i + 1) * batch_size] = features_batch
    labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= sample_count:
      break
  return features, labels

train_features, train_labels = extract_features(train_dir, 6000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1005)

train_features = np.reshape(train_features, (6000, 4 * 4 * 1024))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 1024))
test_features = np.reshape(test_features, (1005, 4 * 4 * 1024))

from keras import models
from keras import layers
from keras import optimizers
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras import optimizers

model = models.Sequential()

model.add(layers.Input(shape=(4 * 4 * 1024,)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))
model.compile(
    optimizers.RMSprop(learning_rate=2e-5),
    loss='categorical_crossentropy',
    metrics=['acc']
)

history = model.fit(
    train_features, train_labels,
    epochs=30,
    batch_size=40,
    validation_data=(validation_features, validation_labels)
)

IMG_WIDTH = 150
IMG_HEIGHT = 150
NUM_CLASSES = 2
from keras import Input, Model

model = models.Sequential()
def model_maker():
  base_model = MobileNet(include_top=False, input_shape = (IMG_WIDTH, IMG_HEIGHT, 3))

  for layer in base_model.layers[:]:
    layer.trainable = False

  input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
  custom_model = base_model(input)
  custom_model = GlobalAveragePooling2D()(custom_model)
  custom_model = Dense(64, activation='relu')(custom_model)
  custom_model = Dropout(0.5)(custom_model)
  predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)
  return Model(inputs=input, outputs=predictions)

model = model_maker()
model.summary()

from keras import optimizers

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='categorical'
)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=1e-5),
              metrics=['acc']
)
train_classes = train_generator.classes
num_classes = np.unique(train_classes).shape[0]
print(f'Количество классов: {num_classes}')

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator
)

import matplotlib.pyplot as plt
def show_history(store):
    acc = store.history['acc']
    val_acc = store.history['val_acc']
    loss = store.history['loss']
    val_loss = store.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'r', label='Точность на обучающей выборке')
    plt.plot(epochs, val_acc, 'bo', label='Точность на проверочной выборке')
    plt.title('График точности на проверочной и обучающей выборках')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Потери на обучающей выборке')
    plt.plot(epochs, val_loss, 'bo', label='Потери на проверочной выборке')
    plt.title('График потерь на проверочной и обучающей выборках')
    plt.legend()
    plt.show()

show_history(history)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('Точность на контрольной выборке:', test_acc)