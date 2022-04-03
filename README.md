# CS6910 ASSIGNMENT 2

Assignment 2 of the CS6910: Fundamentals of Deep Learning Course by Dr. Mitesh Khapra

Project Contributors
- Mayukh Das(CS21S064)
- Amartya Basu(CS21S063)

The github repository contains three folder  which are as follows :
1. Part A : Training from scratch 
2. Part B: Loading and Fine tuning a pre-trained model
3. Part C : Using a pre-trained model as it is YOLOv3 object detection in realtime video

## PART A
In this folder there are 3 notebook(.ipynb) files and 3 .py files.

[cs6910_assignment2_partA_question1_2_3.ipynb](PART%20A/cs6910_assignment2_partA_question1_2_3.ipynb) for code related to Question 1,2,3 of assignment

[cs6910_assignment2_partA_question4.ipynb](PART%20A/cs6910_assignment2_partA_question4.ipynb) for code related to Question 4 of assignment

[cs6910_assignment2_partA_question5.ipynb](PART%20A/cs6910_assignment2_partA_question5.ipynb)  for code related to Question 5 of assignment

[cs6910_assignment2_parta_question1_2_3.py](PART%20A/cs6910_assignment2_parta_question1_2_3.py) for code related to Question 1,2,3 in .py format for running through command line.

[cs6910_assignment2_parta_question4.py](PART%20A/cs6910_assignment2_parta_question4.py) for code related to Question 4 in .py format for running through coomand line.

[cs6910_assignment2_parta_question5_guided_backprop.py](PART%20A/cs6910_assignment2_parta_question5_guided_backprop.py) for code related to Question 5 in .py format for running through command line.

### How to use 
We have used two approaches - jupyter notebook approach and the command line approach

#### Jupyter Notebook Approach
Download the **.ipynb** files in your local systems or google colab. Download all the necessary dependencies- wget, tensorflow, matplotlib etc.
We would recommend to run the **.ipynb** file in google colab to get rid of the local dependencies requirements.

For  running ```cs6910_assignment2_partA_question1_2_3.ipynb``` in jupyter notebook to test over multiple hyperparameter configuration pass a parameter 

to the ```train()``` function.

The parameter is a dictionary and its format is shown below.
```python
    config = {
    "kernel_sizes" : [(3,3),(3,3),(5,5),(3,3),(3,3)],
    "activation" : 'relu',
    "learning_rate": 1e-3,
    "filters_list" : [32,32,64,64,128],
    "dense_layer_size" : 128,
    "batch_normalization": "True",
    "data_augment": "False",
    "weight_decay":0.0005,
    "dropout":0.2,
    "batch_size":64,
    "epochs":3
    }
```
Following are some hyperparameter configurations swept over by wandb
```python
 #Sweep configuration for runs
sweep_config = {
  "name" : "best-sweep",
  "method" : "bayes",
  "metric" : {
      "name" : "val_accuracy",
      "goal" : "maximize"
  },
  "parameters" : {
    "epochs" : {
      "values" : [10,20,30]
    },
    "learning_rate" :{
      "values" : [1e-3,1e-4]
    },
    "kernel_sizes":{
        "values" : [[(3,3),(3,3),(3,3),(3,3),(3,3)],
                    [(3,3),(3,3),(5,5),(7,7),(7,7)],
                    [(11,11),(11,11),(7,7),(5,5),(3,3)],
                    [(3,3),(5,5),(7,7),(9,9),(11,11)],
                    [(5,5),(5,5),(5,5),(5,5),(5,5)]]
    },
    "filters_list":{
        "values" : [[32,32,32,32,32],[256,128,64,32,32],[32,64,64,128,128],[32,64,128,256,512],[64,32,64,32,64]]
    },
    "weight_decay":{
      "values": [0,0.0005,0.005]  
    },
    "data_augment":{
        "values": ["True","False"]
    },
    "batch_size":{
        "values":[32,64]
    },
    "activation":{
        "values": ["relu","elu","swish","gelu"]
    },
      "dropout":{
          "values":[0.0,0.2,0.3]
      },
      "dense_layer_size":{
          "values":[64,128,256,512]
      },
      "batch_normalization":{
          "values":["True","False"]
      }
  }
}
```
####  Command Line Approach

Download the **.py** file in your local system. Make sure your local system has all dependencies like tensorflow, wget, matplotlib etc. Keep in mind that if any of te dependencies are missing the code will not work and show package missing error.Run the below command to install the dependencies 

```
pip install wget
```

```
pip install tensorflow
```
```
pip install zipfile
```

After installing the dependencies run the below command for the assignments.

```
python3 cs6910_assignment2_parta_question1_2_3.py
```
The above command will run the code with default hyperparameter configuration to display the available options to pass as commandline arguments please enter the following command
```
python3 cs6910_assignment2_parta_question1_2_3.py -h
```
After that arguments can be passed like
```
python3 cs6910_assignment2_parta_question1_2_3.py -e 30
```
Will set the epoch to 30 all other hyperparameters are optional when not given will set to default.In the help option all default values of the hyperparameters are displayed.
The above command will run the code and internally invoke the ```train()``` function and a plot of train accuracy validation accuracy and train loss ,validation loss will be saved as .jpg file in the default working directory which can be used for better visualisation.


```
python3 cs6910_assignment2_parta_question4.py
```
In this part no arguments need to be passed as the code internally download and use the best model which was found during wandb sweep **Bayes** search over hyperparameters during training . The test accuracy and test loss will be internally computed again on the same **iNaturalist** dataset which was used in previous part.
Some image plots as mentioned in the Question 4  will be saved as .jpg file and can be viewed for evaluation.
```
python3 cs6910_assignment2_parta_question5_guided_backprop.py
```
This part is very similar to the previous one. 

## PART B
In this folder there is one notebook file and one.py file.

 [cs6910_assignment2_partB_question1_2_3.ipynb](PART%20B/cs6910_assignment2_partB_question1_2_3.ipynb) Part B notebook code.
 
 [cs6910_assignment2_partb_question1_2_3.py](PART%20B/cs6910_assignment2_partb_question1_2_3.py) Part B .py file for command line.
 
 ### How to use
 We have used two approaches for running the code
 
 #### Jupyter Notebook Approach
 Run the **.ipynb** file on your local system or Google Colab. We recommend you to use Colab to remove the package dependecies. Run the code sequentially.
 
 For  running ```cs6910_assignment2_partB_question1_2_3.ipynb``` in jupyter notebook to test over multiple hyperparameter configuration pass a parameter
 
 to the ```train()``` function.
 
The parameter is a dictionary and its format is shown below.

```python
    config = {
    "model": 'InceptionV3',
    "learning_rate": 1e-4,
    "data_augment": "True",
    "dropout":0.2,
    "batch_size":32,
    "fine_tune_last":20,
    "epochs":3
    }
 ```
 Following are some hyperparameter configurations swept over by wandb
 ```python
#Sweep configuration for runs
sweep_config = {
  "name" : "best-sweep-for-pretrained-model",
  "method" : "bayes",
  "metric" : {
      "name" : "val_accuracy",
      "goal" : "maximize"
  },
  
  "parameters" : {
      "model" : {
          "values" : ["InceptionV3", "InceptionResNetV2","Xception","ResNet50","MobileNetV2"]
      },

  "learning_rate" :{
      "values" : [1e-3,1e-4]
  },
  "data_augment" : {
      "values" : ["True","False"]
  },
  "dropout" : {
      "values" : [0.2,0.3,0.4]
  },

  "batch_size" : {
      "values" : [32,64]
  },
  "fine_tune_last" : {
  "values" : [0,10,20,30]
  },
    "epochs" : {
      "values" : [5,10,15,20]
    }
  }
}
 ```
 
 #### Command line approach
 
 Install the following dependencies using the given below command.
  ```
  pip install wget
  ```
  ```
  pip install zipfile
  ```
  
  After running the above commands run the given below command to check the code -
  ```
  python3 cs6910_assignment2_partb_question1_2_3.py
  ```
  The above command will run the code with default hyperparameter configuration. To display the available options to pass as commandline arguments please enter the following command
```
python3 cs6910_assignment2_partb_question1_2_3.py -h
```
After that arguments can be passed like
```
python3 cs6910_assignment2_parta_question1_2_3.py -e 30 -m InceptionResNetV2
```
Will set the epochs to 30 and model to InceptionResNetV2 for training and  all other hyperparameters are optional when not given will set to default.In the help option all default values of the hyperparameters are displayed.
The above command will run the code and internally invoke the ```train()``` function and a plot of train accuracy validation accuracy and train loss ,validation loss will be saved as .jpg file in the default working directory which can be used for better visualisation.
Test accuracy over test dataset is also calculated and displayed.

  If there are any problems while running the code in terminal kindly use the below command to check the manual
  
  ```
  python3 -h
  ```
  
  
  **Note: Initially the iNaturalist dataset will be downloaded in current working directory so it may take some time depending on the internet speed. We recommend you to use the same working directory for the above 3 files to avoid redundant downloads**
 
## PART C 
This project implements video object detection classifier using pretrained yolov3 models. 
The yolov3 models are taken from the official yolov3 paper which was released in 2018. The yolov3 implementation is from [darknet](https://github.com/pjreddie/darknet). Also, this project implements an option to perform classification real-time using the webcam.
 
### Problem Statement

We have heard news of various fire in appartments, office, colony etc. Most of the times the fire fighters are not able to distinguish between objects and humans in that environment. This happens mostly because of the smoke, dust etc. This can cause loss of human life in that scenario.

### Our solution

We have used YoloV3 for object detection in such environments. Our code can properly identify humans, cars etc in those kind of environments. Since this just a basic work for this assignment. We hope that we will carry on our work so that it can properly identify humans in not only fire incidents but also in places like floods, eartquakes, landslides etc.

### How to use?

1) Download the folder

2) Move to the directory
```
cd PART C_OBJECT DETECTION
```

3) To infer on a video that is stored on your local machine
```
python3 yolo.py --video-path='/path/to/video/'
```

Note: This works considering you have the `weights` and `config` files at the yolov3-coco directory.
<br/>
If the files are located somewhere else then mention the path while calling the `yolov3.py`. For more details
```
yolo.py --help
```


### Inference on Video

(https://youtu.be/9U6GWO-uX2k)


## References

1) [Deep Learning lectures by Professor Mitesh Khapra, IIT Madras.](https://youtube.com/playlist?list=PLEAYkSg4uSQ1r-2XrJ_GBzzS6I-f8yfRU)
2) [PyImageSearch YOLOv3 Object Detection with OpenCV Blog](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
3) [https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV](https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV)
4) [https://www.tensorflow.org](https://www.tensorflow.org)
5) [https://towardsdatascience.com/how-to-visually-explain-any-cnn-based-models-80e0975ce57](https://towardsdatascience.com/how-to-visually-explain-any-cnn-based-models-80e0975ce57)
6) [https://matplotlib.org](https://matplotlib.org)

## License

The code in this project is made by Amartya and Mayukh.
