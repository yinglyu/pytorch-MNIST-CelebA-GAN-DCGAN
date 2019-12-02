
# coding: utf-8

# In[1]:


import os, time
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets.mnist as mnist
import numpy 
import skimage
import cv2


if not os.path.isdir('MNIST_DCGAN_results'):
    os.mkdir('MNIST_DCGAN_results')
if not os.path.isdir('MNIST_DCGAN_results/RGB'):
    os.mkdir('MNIST_DCGAN_results/RGB')
if not os.path.isdir('MNIST_DCGAN_results/RGB/mnist'):
    os.mkdir('MNIST_DCGAN_results/RGB/mnist')
test_torch = mnist.read_image_file(os.path.join('data/raw/','t10k-images-idx3-ubyte'))
# test_torch.size()
tor2numpy=test_torch.numpy()
# tor2numpy.shape
# %matplotlib inline
# plt.figure(figsize=(64,64),dpi=1)
# plt.imshow(tor2numpy[0], cmap='gray')

for i in range(0,1000):
    #dst=transform.resize(tor2numpy[i], (64, 64))
    #misc.imsave('MNIST_DCGAN_results/mnist/'+str(i)+".jpg",dst)
    src=skimage.transform.resize(tor2numpy[i], (64, 64))
    imageio.imwrite("temp.jpg",skimage.img_as_ubyte(src))
    src = cv2.imread("temp.jpg", 0)
    src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    cv2.imwrite('MNIST_DCGAN_results/RGB/mnist/'+str(i)+".jpg", src_RGB)


print("finished mnist export")