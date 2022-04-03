# Guided Backpropagation
import matplotlib.pyplot as plt
import numpy as np
import random
import wget
import os
from zipfile import ZipFile
from PIL import Image
from matplotlib import gridspec

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

model_url = 'https://drive.google.com/uc?id=10QlusHKjLKjV1aAELCTo73bJzWdZ6px5&export=download'
if not os.path.exists('model-guidedback.h5'):
  model_filename = wget.download(model_url)
else:
  model_filename = 'model-guidedback.h5'
# Recreate the exact same model, including its weights and the optimizer
model = keras.models.load_model(model_filename)
# Show the model architecture
model.summary()

test_generator = ImageDataGenerator(dtype=tf.float32,validation_split=0.0,data_format='channels_last').flow_from_directory(
        'inaturalist_12K/val',
        target_size = image_size,
        batch_size = 200,
        color_mode = 'rgb',
        class_mode = 'sparse',
        shuffle=False,
        seed=123
        )
test_generator_ = test_generator
images,labels = next(iter(test_generator_))
class_names =list(test_generator.class_indices)

#Sampling a random image from each class
images=[]
labels=[]
pred_labels = []
for count in range(len(class_names)):
    Images,Labels = next(iter(test_generator))
    for Image,Label in random.sample(list(zip(Images,Labels)),1):
        images.append(Image)
        labels.append(Label.astype('int'))
        Image_scaled = np.expand_dims(Image,axis=0)
        pred_labels.append(tf.argmax(model.predict(Image_scaled),1).numpy()[0])

#Creating custom gradient for guided back prop
@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

#Overridimg the activation of required layers with our custom activation
layer_activation_list = [layer.activation for layer in model.layers if hasattr(layer,'activation')]
activation_model = tf.keras.models.Model([model.inputs],[model.layers[9].output]) #model.layers[9] is the 'CONV5' layer
for activation in layer_activation_list:
  activation = guidedRelu

with tf.GradientTape() as tape:
    inputs = tf.cast(images,tf.float32)
    tape.watch(inputs)
    outputs = activation_model(inputs)
grads = tape.gradient(outputs,inputs)

#Guided Backprop visuals
plt.figure(figsize=(200,200))
plt.title("Activations of CONV5 layer")
display = np.zeros((10,1))
index = random.sample(range(outputs.numpy()[0,:,:,0].flatten().shape[0]),10)
# print(index)
g = gridspec.GridSpec(3,10,hspace=0.0,wspace=0.1,left=0.8,right=0.9,top=0.9,bottom=0.8)
for j in range(10):
    ax = plt.subplot(g[0,j])
    ax.imshow(images[j].astype("uint8"))
    ax.set_title(class_names[labels[j].astype("int")])
    ax.axis("off")
    ac = outputs.numpy()[j,:,:,0].flatten()
    for m in range(10):
        for n in range(1):
            display[m,n] = ac[index[m]]
    ax = plt.subplot(g[1,j])
    ax.imshow(display,cmap='gray')
    gb_viz = grads[j]
    gb_viz = np.dstack((
                gb_viz[:, :, 0],
                gb_viz[:, :, 1],
                gb_viz[:, :, 2],
            ))       
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    ax = plt.subplot(g[2,j])
    ax.imshow(gb_viz)
    ax.axis("off")
plt.savefig("guided_backprop.jpg",bbox_inches="tight")