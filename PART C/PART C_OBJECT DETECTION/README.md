# YOLOv3-Object-Detection-with-OpenCV

This project implements video object detection classifier using pretrained yolov3 models. 
The yolov3 models are taken from the official yolov3 paper which was released in 2018. The yolov3 implementation is from [darknet](https://github.com/pjreddie/darknet). Also, this project implements an option to perform classification real-time using the webcam.
 
##Problem Statement

We have heard news of various fire in appartments, office, colony etc. Most of the times the fire fighters are not able to distinguish between objects and humans in that environment. This happens mostly because of the smoke, dust etc. This can cause loss of human life in that scenario.

## Our solution

We have used YoloV3 for object detection in such environments. Our code can properly identify humans, cars etc in those kind of environments. Since this just a basic work for this assignment. We hope that we will carry on our work so that it can properly identify humans in not only fire incidents but also in places like floods, eartquakes, landslides etc.

## How to use?

1) Download the folder

2) Move to the directory
```
cd PART C_OBJECT DETECTION
```

3) To infer on a video that is stored on your local machine
```
python3 yolo.py --video-path='/path/to/video/'
```
5) To infer real-time on webcam
```
python3 yolo.py
```

Note: This works considering you have the `weights` and `config` files at the yolov3-coco directory.
<br/>
If the files are located somewhere else then mention the path while calling the `yolov3.py`. For more details
```
yolo.py --help
```


## Inference on Video

(https://youtu.be/9U6GWO-uX2k)


## References

1) Deep Learning lectures by Professor Mitesh Khapra, IIT Madras.
2) [PyImageSearch YOLOv3 Object Detection with OpenCV Blog](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
3) [https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV]

## License

The code in this project is made by Amartya and Mayukh. 
