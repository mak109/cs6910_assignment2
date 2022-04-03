# Loading and Fine-tuning pretrained Models
import matplotlib.pyplot as plt
import numpy as np
import random
import wget
import os
import datetime
from zipfile import ZipFile
from PIL import Image
from inspect import *
from matplotlib import gridspec

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential,regularizers,optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import * 
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
train_dir = 'inaturalist_12K/train'
test_dir = 'inaturalist_12K/val'

image_size = (256,256)
num_classes = 10

import argparse
#default config
config = {
    "model": 'InceptionV3',
    "learning_rate": 1e-4,
    "data_augment": True,
    "dropout":0.2,
    "batch_size":32,
    "fine_tune_last":0,
    "epochs":3
    }
parser = argparse.ArgumentParser(description='Process the hyperparameters.')
parser.add_argument('-e','--epochs', type=type(config['epochs']), nargs='?', default = config['epochs']
                    ,help=f"Number of epochs(default {config['epochs']})")
parser.add_argument('-m','--model',type=type(config['model']),nargs='?',default=config['model'],help='Model to be used for pretraining and finetuning(default InceptionV3)')
parser.add_argument('-lr','--learning_rate', type=type(config['learning_rate']), nargs='?', default = config['learning_rate']
                    ,help=f"Learning rate of the model default( {config['learning_rate']}")
parser.add_argument('-d','--dropout', type=type(config['dropout']), nargs='?', default = config['dropout']
                    ,help=f"Dropout added to the dense layer(default {config['dropout']})")
parser.add_argument('-da','--data_augment', type=type(config['data_augment']), nargs='?', default = config['data_augment']
                    ,help=f"takes value True/False for using Data augmentation or not (default {config['data_augment']})")
parser.add_argument('-bs','--batch_size', type=type(config['batch_size']), nargs='?', default = config['batch_size']
                    ,help=f"Batch Size to be used(default {config['batch_size']})")
parser.add_argument('-ftl','--fine_tune_last', type=type(config['fine_tune_last']), nargs='?', default = config['fine_tune_last']
                    ,help=f"Fine Tune last k layers where k is passed as input (default {config['fine_tune_last']})")
parser.add_argument('-cp','--checkpointing',type=bool,nargs='?',default=False,help=f"Optional argument for model check pointing( default {False})")

args = parser.parse_args()
config  = vars(args)
print(config)
#Creating dictionary of models based on imagenet 
model_list = dict()
for key,value in getmembers(tf.keras.applications,isfunction):
    model_list[key] = value

#Creating model using pretrained model
def CNN(config):
    base_model = model_list[config['model']](input_shape=image_size +(3,),include_top=False,weights='imagenet')
    base_model.trainable = True #this is important
    if(len(base_model.layers) > config['fine_tune_last']):
        for layer in base_model.layers[:-config['fine_tune_last']]:
            layer.trainable = False    
    global_average_layer = layers.GlobalAveragePooling2D()
    prediction_layer = layers.Dense(num_classes,activation='softmax')
    inputs = layers.Input((image_size[0],image_size[1],3))
    input_rescale=layers.Rescaling(1./255)(inputs)
    x = base_model(input_rescale)    
    x = global_average_layer(x)
    x = layers.Dropout(config['dropout'])(x)
    outputs = prediction_layer(x)
    model = keras.Model(inputs,outputs)
    return model

def train():
    val_generator = ImageDataGenerator(dtype=tf.float32,validation_split=0.1,data_format='channels_last').flow_from_directory(
        train_dir,
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
    train_dir,
    target_size = image_size,
    batch_size = config['batch_size'],
    color_mode = 'rgb',
    class_mode = 'sparse',
    shuffle=True,
    subset='training',
    seed=123
    )
    
    model = CNN(config)
    
    model.compile(
    optimizer=optimizers.Adam(learning_rate=config["learning_rate"]),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )
    
        #For checkpointing default value is False
    if config['checkpointing']:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, f'models_pretrained_{datetime.datetime.now()}')
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
    return history,model

history,model = train()

# Testing
test_generator = ImageDataGenerator(dtype=tf.float32,validation_split=0.0,data_format='channels_last').flow_from_directory(
        'inaturalist_12K/val',
        target_size = image_size,
        batch_size = 200,
        color_mode = 'rgb',
        class_mode = 'sparse',
        shuffle=False,
        seed=123
        )
loss0, accuracy0 = model.evaluate(test_generator)

#Visualization
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
plt.savefig('metrics-pretrained.jpg')
plt.show()

'''This section is used for loading the models saved with datetime when checkpointing is True'''
# #This can be used when checkpointing is set to True and models are saved in model directory with proper name in the current working directory
# model_dir = 'models_pretrained_2022-04-03 00:00:29.823768' #model director name goes here
# new_model = tf.keras.models.load_model(model_dir)
# # Check its architecture
# new_model.summary()
