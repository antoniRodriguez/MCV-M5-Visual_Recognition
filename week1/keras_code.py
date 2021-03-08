
from keras.applications.resnet50 import ResNet50 as RN50
from keras.preprocessing import image
from keras import models
from keras import layers,Input
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras import optimizers

end='keras_model'

BOARD_PATH = '../outs/'
EXPERIMENT_NAME = 'keras_training'

train_data_dir='/home/mcv/datasets/MIT_split/train'
val_data_dir='/home/mcv/datasets/MIT_split/test'
test_data_dir='/home/mcv/datasets/MIT_split/test'
img_width = 254
img_height=254
batch_size=32
number_of_epoch=20
validation_samples=807


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    assert dim_ordering in {'channels_first', 'channels_last'}

    if dim_ordering == 'channels_first':
        # 'RGB'->'BGR'
        x = x[ ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= 103.939
        x[ 1, :, :] -= 116.779
        x[ 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x
    
	
# create the base pre-trained model
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(254,254,3)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, (3,3),activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, (3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3),activation='relu'))
    #model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(256, (3,3),activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, (3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3),activation='relu'))
    #model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, (3,3),activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, (3,3),activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(8, activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='Adamax', metrics=['accuracy'])

    return model

model = create_model()

with open(f'../model_{EXPERIMENT_NAME}.txt','w') as file:
	model.summary(print_fn=lambda x:file.write(x + '\n'))

plot_model(model, to_file=f'../model_{EXPERIMENT_NAME}.png', show_shapes=True, show_layer_names=True)

    
for layer in model.layers:
    print(layer.name, layer.trainable)

#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
	  preprocessing_function=preprocess_input,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

tbCallBack = TensorBoard(log_dir=BOARD_PATH+EXPERIMENT_NAME, histogram_freq=0, write_graph=True)
history=model.fit_generator(train_generator,
        steps_per_epoch=(int(400//batch_size)+1),
        epochs=number_of_epoch,
        validation_data=validation_generator,
        validation_steps= (int(validation_samples//batch_size)+1), callbacks=[tbCallBack])


result = model.evaluate_generator(test_generator, int(validation_samples//batch_size))
print(result)

#saving model
model.save(f'../model_{EXPERIMENT_NAME}.h5')

# list all data in history

if False:
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('accuracy.jpg')
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('loss.jpg')
