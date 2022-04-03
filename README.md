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
In this folder there are 3 notebook files and 3 .py files.

[cs6910_assignment2_partA_question1_2_3.ipynb](PART%20A/cs6910_assignment2_partA_question1_2_3.ipynb) for code related to Question 1,2,3 of assignment

[cs6910_assignment2_partA_question4.ipynb](PART%20A/cs6910_assignment2_partA_question4.ipynb) for code related to Question 4 of assignment

[cs6910_assignment2_partA_question5.ipynb](PART%20A/cs6910_assignment2_partA_question5.ipynb)  for code related to Question 5 of assignment

[cs6910_assignment2/PART A/cs6910_assignment2_parta_question1_2_3.py](PART%20A/cs6910_assignment2_parta_question1_2_3.py) for code related to Question 1,2,3 in .py format for running through command line.

[cs6910_assignment2/PART A/cs6910_assignment2_parta_question4.py](PART%20A/cs6910_assignment2_parta_question4.py) for code related to Question 4 in .py format for running through coomand line.

[cs6910_assignment2/PART A/cs6910_assignment2_parta_question5_guided_backprop.py](PART%20A/cs6910_assignment2_parta_question5_guided_backprop.py) for code related to Question 5 in .py format for running through command line.

### How to use 
We have used two approaches - jupyter notebook approach and the command line approach

#### Jupyter Notebook Approach
Download the .ipynb files in your local systems or google colab. Dowload all the necessary dependencies- wget, tensorflow, matplotlib etc.
We would recommend to run the .ipynb file in google colab to get rid of the local dependencies requirements.

####  Command Line Approach

Download the .py file in your local system. Make sure your local system has all dependencies like tensorflow, wget, matplotlib etc. Keep in mind that if any of te dependencies are missing the code will not work and show package missing error.

## PART B
In this folder there is one notebook file and one.py file.

 
 
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

1) Deep Learning lectures by Professor Mitesh Khapra, IIT Madras.
2) [PyImageSearch YOLOv3 Object Detection with OpenCV Blog](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
3) [https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV]

## License

The code in this project is made by Amartya and Mayukh. 
