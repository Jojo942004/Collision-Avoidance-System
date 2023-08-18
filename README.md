# Collision-Avoidance-System


May I present you a collision-avoidance-system using the DepthAI OAK-D

## Description

For additional information to the project check out the detailed paper I wrote. Please also check out the added code, dataset, trained models, PowerPoint presentation and additional files.
https://drive.google.com/drive/folders/1P4aJD1bQrliQuO2XWLKK_LW5oQck8lYP?usp=share_link

## Demo - YoloV5 COCO dataset


https://user-images.githubusercontent.com/38126993/224107647-abcb29aa-e71e-4b83-9bcb-76eba793be66.mp4

Object Detection,
Distance Messurement and Path Finding



## Getting Started

1. Clone the repo

```
git clone https://github.com/Jojo942004/Collision-Avoidance-System
```
2. download the folders from google drive to get access to custom models trained on boat images, pretrained models of the yolov5 repo, custom boat dataset

https://drive.google.com/drive/folders/1P4aJD1bQrliQuO2XWLKK_LW5oQck8lYP?usp=share_link

3. connect your OAK-D by Luxonis
4. simply run 

```
python collision-avoidance-system.py
```
## Additional Resources
To gain access to trial data, custom models please visit  
https://drive.google.com/drive/folders/1P4aJD1bQrliQuO2XWLKK_LW5oQck8lYP?usp=share_link

## Object Detection in action


https://user-images.githubusercontent.com/38126993/224103241-811ba76e-fdea-41a4-99da-8f456bd0a4cc.mp4

## Presentation


[Presentation Collision Avoidance System.pdf](https://github.com/Jojo942004/Collision-Avoidance-System/files/10934253/Presentation.Collision.Avoidance.System.pdf)

### Dependencies
depthai==2.17.4.0  
torch==1.12.0+cu116  
torchaudio==0.12.0+cu116  
torchvision==0.13.0+cu116  
opencv-contrib-python==4.6.0.66   
opencv-python==4.6.0.66  
numpy  
math  
multiprocessing  
time  


## Authors

Johannes Zimmermann
[@Jojo942004](https://github.com/Jojo942004)


