"""
# Visualization part
  In this part we find the test accuracy on the best model found after Question 3 and we did the following
   - Found the test accuracy
   - Plotted an image grid of 10X3 showing true vs predicted label based on best model
   - Visualize the filters of 1st Convolution layer
"""
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
from keras.preprocessing.image import ImageDataGenerator
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

model_url = 'https://drive.google.com/uc?id=1t9_xgJkFr7j5v3rOXAuJAN7mFCDpa4Dk&export=download'
if not os.path.exists('model-best.h5'):
  model_filename = wget.download(model_url)
else:
  model_filename = 'model-best.h5'
# Recreate the exact same model, including its weights and the optimizer
model = keras.models.load_model(model_filename)
# Show the model architecture
model.summary()

"""## Test Accuracy on best model loaded above"""

test_generator = ImageDataGenerator(dtype=tf.float32,validation_split=0.0,data_format='channels_last').flow_from_directory(
        'inaturalist_12K/val',
        target_size = image_size,
        batch_size = 200,
        color_mode = 'rgb',
        class_mode = 'sparse',
        shuffle=False,
        seed=123
        )
test_loss,test_acc = model.evaluate(test_generator)
print(f"Test Accuracy : {test_acc} Test loss : {test_loss}")
class_names =list(test_generator.class_indices)

#Sampling from test data for visualization part
images=[]
labels=[]
pred_labels = []
for count in range(len(class_names)):
    Images,Labels = next(iter(test_generator))
    for Image,Label in random.sample(list(zip(Images,Labels)),3):
        images.append(Image)
        labels.append(Label.astype('int'))
        Image_scaled = np.expand_dims(Image,axis=0)
        pred_labels.append(tf.argmax(model.predict(Image_scaled),1).numpy()[0])

#Printing number of correct labels
np.sum(np.array(labels)==pred_labels)

"""## Image grid of 10X3 showing true vs predicted label based on best model"""

idx=0
fig = plt.figure(figsize=(60,60))
g = gridspec.GridSpec(10,3,wspace=0.1,left=0.7,right=0.9)
for i in range(10):
    for j in range(3):
        ax = plt.subplot(g[i,j])
        ax.imshow(images[idx].astype("uint8"))
        title = f"True label : {class_names[labels[idx].astype('int')]}\nPredicted label : {class_names[pred_labels[idx]]}"
        ax.set_title(title)
        ax.axis("off")
        idx += 1
plt.savefig("test_image_37percent.jpg",bbox_inches="tight")

"""## Visualization of filters"""

#Random Image 
test_generator.reset()
images,labels = next(iter(test_generator))
rand_indx = random.randint(0,200)
random_image = images[rand_indx].astype("uint8")
random_image_label = labels[rand_indx].astype("int")
filters_activation = model.layers[1].call(images).numpy()[rand_indx]
filters = model.layers[1].weights[0].numpy()
plt.figure(figsize=(5,5))
plt.imshow(random_image)
plt.axis("off")
plt.title("Random image label : "+str(class_names[random_image_label]))
plt.savefig("random_image_best_model.jpg",bbox_inches="tight")


#Filters activation for the above random image
idx=0
fig = plt.figure(figsize=(60,60))
# fig.tight_layout()
g = gridspec.GridSpec(4,8,wspace=0.1,left=0.7,right=0.9,top=0.7,bottom=0.6)
for i in range(4):
    for j in range(8):
        ax = plt.subplot(g[i,j])
        ax.imshow(filters_activation[:,:,idx].astype("uint8"))
        title = "filter : "+str(idx+1)
        ax.set_title(title)
        ax.axis("off")
        idx += 1
plt.savefig("random_image_best_model_filters_activation.jpg",bbox_inches="tight")

#Filters for the above random image
idx=0
fig = plt.figure(figsize=(60,60))
g = gridspec.GridSpec(32,3,wspace=0.2,hspace=0.5,left=0.8,right=0.9)
for i in range(32):
    for j in range(3):
        ax = plt.subplot(g[i,j])
        ax.imshow(filters[j,:,:,i],cmap='gray')
        title = f"filter : {i+1} channel : {j+1}"
        ax.set_title(title)
        ax.axis("off")
        idx += 1
plt.savefig("random_image_best_model_filters.jpg",bbox_inches="tight")
