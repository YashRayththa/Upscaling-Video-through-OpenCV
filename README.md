# Upscaling-Video-through-OpenCV
## Import the following libraries & functions:  
import subprocess  
from moviepy.editor import VideoFileClip  
import os.path as osp  
import glob  
import cv2  
import numpy as np  
import torch  
import functools  
import torch.nn as nn  
import torch.nn.functional as F    
import timeit  
from tensorflow.keras.layers import BatchNormalization  
from cv2 import dnn_superres  
import tensorflow as tf  
import os  
import matplotlib.pyplot as plt  
from PIL import Image  
import keras.applications  
from tensorflow.python.keras.layers import Add, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda  
from tensorflow.python.keras.models import Model  
from keras.applications import VGG19  

## Working:  
1. The code takes input as mp4 files and converts each frame into a image.
2. The images are then individually upscaled using ESRGAN model from:https://github.com/xinntao/ESRGAN
3. The audio is extarcted from mp4 file using moviepy.editor and stored seperately
4. The upscaled images are compressed into a video after sorting.
5. The audio file is added to the output video.
6. The video is upscaled and ready to Go.

## Advantages of this program:  
The model works efficiently 
The resolution significantly improves through this program

## Drawbacks:
This requires enormous processing time and GPU power

## The following screenshots show the significant improvement in resolution of the initial and output videos:  

![Screenshot 2023-12-11 100800](https://github.com/YashRayththa/Upscaling-Video-through-OpenCV/assets/87801719/a90c064f-d68a-4862-b317-aff05f0a83ce)
![Screenshot 2023-12-11 100828](https://github.com/YashRayththa/Upscaling-Video-through-OpenCV/assets/87801719/520868e1-e95f-45b2-afba-62108775c764)


## GoogleDrive:
The following Google drive contains test videos, output videos, ESRGAN model & python files:
  https://drive.google.com/drive/folders/1_Ty5arZtrQtL4zIYIYvUo3nhLQ_2TF7v?usp=sharing
