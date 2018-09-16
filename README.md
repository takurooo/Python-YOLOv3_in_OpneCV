# Python-YOLO_in_OpneCV
Execute pretrained YOLOv3 in OpneCV.  

![image](https://user-images.githubusercontent.com/35373553/45593378-5d00e800-b9c0-11e8-95f0-2d80b9411367.jpg)

# Requirement
- OpneCV 3.4.2
- numpy  


# Installation
Sample code need yolov3.cfg, yolov3.weights and coco.names.  

- yolov3.weights
```
wget https://pjreddie.com/media/files/yolov3.weights
```
- yolov3.cfg / coco.names  
[darknet-github](https://github.com/pjreddie/darknet)
```
darknet_root/  
    　│
    　├ data/  
    　│　└ coco.names
    　├ cfg/  
    　│　└ yolov3.cfg
```

# Usage
```
python main_img.py <image_file_path>
```
Show image with BBox and save file to current directory.

```
python main_video.py <video_file_path>
```
Show video with BBox.


```
python main_webcam.py
```
Activate your webcam and detect object using images through a lens.
