# -*- coding: utf-8 -*-
"""Video_upscaling model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nT-RRs4Yut_ZWUlsBAnONbTq9KnUujGE
"""

!git clone https://github.com/xinntao/ESRGAN

import cv2
from cv2 import dnn_superres
import numpy as np
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras.applications
from tensorflow.python.keras.layers import Add, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.python.keras.models import Model
from keras.applications import VGG19

cam = cv2.VideoCapture("/content/test1.mp4")
fps = cam.get(cv2.CAP_PROP_FPS)
print(fps)

try:


    if not os.path.exists('data'):
        os.makedirs('data')

except OSError:
    print ('Error: Creating directory of data')
currentframe = 0
arr_img = []

while(True):

    ret,frame = cam.read()

    if ret:
        name = './data/frame' + str(currentframe).zfill(3) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        currentframe += 1
        arr_img.append(name)
    else:
        break

import timeit
from tensorflow.keras.layers import BatchNormalization

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

from moviepy.editor import VideoFileClip

video_file = VideoFileClip("/content/test1.mp4")

audio_file = video_file.audio

audio_file.write_audiofile("/content/test1.wav")

print("Audio successfully extracted!")

import os.path as osp
import glob
import cv2
import numpy as np
import torch

model_path = '/content/ESRGAN/models/RRDB_ESRGAN_x4.pth'
device = torch.device('cuda')


test_img_folder = '/content/data/*'

model = RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('/content/upscaled/{:s}.png'.format(base), output)

import cv2
import os

image_folder = "/content/upscaled"

video_name = "/content/output1.mp4"

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))

for image in images:
    image = cv2.imread(os.path.join(image_folder, image))
    video_writer.write(image)

video_writer.release()

print("Video created successfully!")

import subprocess

video_file = "/content/output1.mp4"
audio_file = "/content/test1.wav"
output_file = "/content/final1.mp4"

command = f"ffmpeg -i {video_file} -i {audio_file} -map 0:v -map 1:a -c:v copy -c:a aac -strict experimental {output_file}"

subprocess.call(command, shell=True)

print("Video with added audio successfully created!")

from moviepy.editor import VideoFileClip

video_file_path = "/content/test1.mp4"

video_clip = VideoFileClip(video_file_path)

fps = video_clip.fps
width, height = video_clip.size
duration = video_clip.duration
audio = video_clip.audio

print(f"FPS: {fps}")
print(f"Resolution: {width}x{height}")
print(f"Duration: {duration} seconds")

video_clip.close()

from moviepy.editor import VideoFileClip

video_file_path = "/content/output1.mp4"

video_clip = VideoFileClip(video_file_path)

fps = video_clip.fps
width, height = video_clip.size
duration = video_clip.duration
audio = video_clip.audio

print(f"FPS: {fps}")
print(f"Resolution: {width}x{height}")
print(f"Duration: {duration} seconds")

video_clip.close()