# YOLOv3-Object-Detection-with-OpenCV

This project implements video object detection classifier using pretrained yolov3 models. 
The yolov3 models are taken from the official yolov3 paper which was released in 2018. The yolov3 implementation is from [darknet](https://github.com/pjreddie/darknet). Also, this project implements an option to perform classification real-time using the webcam.

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

[![yolov3-video](https://user-images.githubusercontent.com/26242097/48851021-0785f480-edd0-11e8-8ce4-cdfb78e8a849.png)](https://www.youtube.com/watch?v=AzmCYs5fAn0)
<small> Click on the image to Play the video on YouTube </small>


## References

1) [PyImageSearch YOLOv3 Object Detection with OpenCV Blog](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
2) [https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV]

## License

The code in this project is made by Amartya and Mayukh. 
