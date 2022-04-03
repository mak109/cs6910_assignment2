# Training CNN from scratch
import matplotlib.pyplot as plt
import numpy as np
import random
import wget
import os
import datetime
from zipfile import ZipFile
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential,regularizers,optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
random.seed(123)

url='https://storage.googleapis.com/wandb_datasets/nature_12K.zip'
filename = os.path.basename(url)

if not os.path.exists(filename) and not os.path.exists("inaturalist_12K"):
  filename = wget.download(url)
  with ZipFile(filename, 'r') as z:
    print('Extracting all the files now...')
    z.extractall()
    print('Done!')
  os.remove(filename)

image_size = (256,256)
num_layers = 5
num_classes = 10

import argparse
#default config
config = {
    "kernel_sizes" : [3,3,5,3,3],
    "activation" : 'relu',
    "learning_rate": 1e-3,
    "filters_list" : [32,32,64,64,128],
    "dense_layer_size" : 128,
    "batch_normalization": True,
    "data_augment": False,
    "weight_decay":0.0005,
    "dropout":0.2,
    "batch_size":64,
    "epochs":3
    }
parser = argparse.ArgumentParser(description='Process the hyperparameters.')
parser.add_argument('-e','--epochs', type=type(config['epochs']), nargs='?', default = config['epochs']
                    ,help=f"Number of epochs(default {config['epochs']})")
parser.add_argument('-ac','--activation', type=type(config['activation']), nargs='?', default = config['activation']
                    ,help=f"Activation to be used for every layer( default {config['activation']})")
parser.add_argument('-lr','--learning_rate', type=type(config['learning_rate']), nargs='?', default = config['learning_rate']
                    ,help=f"Learning rate of the model default( {config['learning_rate']}")
parser.add_argument('-fl','--filters_list', type=int,nargs=5, default = config['filters_list']
                    ,help=f"Number of filters to be used on each convolution layer must be of length 5(default {config['filters_list']})")
parser.add_argument('-bn','--batch_normalization', type=type(config['batch_normalization']), nargs='?', default = config['batch_normalization']
                    ,help=f"takes value True/False for using batch normalization or not default( {config['batch_normalization']})")
parser.add_argument('-d','--dropout', type=type(config['dropout']), nargs='?', default = config['dropout']
                    ,help=f"Dropout added to the dense layer(default {config['dropout']})")
parser.add_argument('-da','--data_augment', type=type(config['data_augment']), nargs='?', default = config['data_augment']
                    ,help=f"takes value True/False for using Data augmentation or not (default {config['data_augment']})")
parser.add_argument('-wd','--weight_decay', type=type(config['weight_decay']), nargs='?', default = config['weight_decay']
                    ,help=f"Weight decay for L2 regularization(default {config['weight_decay']})")
parser.add_argument('-bs','--batch_size', type=type(config['batch_size']), nargs='?', default = config['batch_size']
                    ,help=f"Batch Size to be used(default {config['batch_size']})")
parser.add_argument('-ds','--dense_layer_size', type=type(config['dense_layer_size']), nargs='?', default = config['dense_layer_size']
                    ,help=f"Number of neurons in fully connected dense layer( default {config['dense_layer_size']})")
parser.add_argument('-ks','--kernel_sizes', type=int, nargs=5, default = config['kernel_sizes']
                    ,help=f"Kernel sizes to be used in each of five layers default( {config['kernel_sizes']} )")
parser.add_argument('-cp','--checkpointing',type=bool,nargs='?',default=False,help='Optional argument for model check pointing default value is False')

args = parser.parse_args()
config  = vars(args)
config['kernel_sizes'] = [(x,x) for x in args.kernel_sizes]
print(config)
def CNN(config):
    model = Sequential([
        layers.Input((image_size[0],image_size[1],3)),
        layers.Rescaling(1./255)
        ])
    
    for l in range(num_layers):
        model.add(layers.Conv2D(filters=config["filters_list"][l],kernel_size=(config["kernel_sizes"][l][0],config["kernel_sizes"][l][1]),
                        activation=config["activation"],padding="same",kernel_regularizer=regularizers.l2(config["weight_decay"])))
        if config["batch_normalization"]:
            model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(config["dense_layer_size"],activation=config["activation"],kernel_regularizer=regularizers.l2(config["weight_decay"])))
    model.add(layers.Dropout(config["dropout"]))

    model.add(layers.Dense(num_classes,activation="softmax"))
    return model
#Training goes here
def train():
    
    val_generator = ImageDataGenerator(dtype=tf.float32,validation_split=0.1,data_format='channels_last').flow_from_directory(
        'inaturalist_12K/train',
        target_size = image_size,
        batch_size = config['batch_size'],
        color_mode = 'rgb',
        class_mode = 'sparse',
        shuffle=True,
        subset='validation',
        seed=123
    
    )
    #Data Augmentation
    if config["data_augment"]:
        data_generator = ImageDataGenerator(
        rotation_range=50, #random rotation between -50(clockwise) to 50(anti-clockwise) degree
        brightness_range=(0.2,0.8), 
        zoom_range=0.3, #zoom in range from [0.7,1.3]
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1, #Horizontal Shifting as a ratio of width
        height_shift_range=0.2,#Vertical Shifting as a ratio of height
        data_format='channels_last',
        validation_split=0.1,
        dtype=tf.float32
        )
    else:
        data_generator = ImageDataGenerator(
            data_format='channels_last',
            validation_split=0.1,
            dtype=tf.float32
        )
    #Train set creation after conditional augmentation
    train_generator = data_generator.flow_from_directory(
    'inaturalist_12K/train',
    target_size = image_size,
    batch_size = config['batch_size'],
    color_mode = 'rgb',
    class_mode = 'sparse',
    shuffle=True,
    subset='training',
    seed=123
    )
    #Building Model based on config 
    model = CNN(config)
    
    #Compiling model 
    model.compile(
    optimizer=optimizers.Adam(learning_rate=config["learning_rate"]),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )
    #For checkpointing default value is False
    if config['checkpointing']:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, f'models_{datetime.datetime.now()}')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        checkpoint_filepath = final_directory
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
          filepath=checkpoint_filepath,
          save_weights_only=False,
          monitor='val_accuracy',
          mode='max',
          save_best_only=True)
          #Fitting Model
        history = model.fit(train_generator,
          validation_data=val_generator,
          epochs=config["epochs"],
          verbose=1,
          callbacks = [model_checkpoint_callback] #Custom callback for checkpointing
          )
    else:
        history = model.fit(train_generator,
          validation_data=val_generator,
          epochs=config["epochs"],
          verbose=1,
          )
    return history

history = train()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('metrics.jpg')
plt.show()

'''This section is used for loading the models saved with datetime when checkpointing is True'''
# #This can be used when checkpointing is set to True and models are saved in model directory with proper name in the current working directory
# model_dir = 'models_2022-04-03 00:00:29.823768' #model director name goes here
# new_model = tf.keras.models.load_model(model_dir)
# # Check its architecture
# new_model.summary()

