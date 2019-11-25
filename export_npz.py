
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
from skimage import transform,data
from scipy import misc


# In[2]:



# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# In[3]:



if not os.path.isdir('MNIST_DCGAN_results'):
    os.mkdir('MNIST_DCGAN_results')
if not os.path.isdir('MNIST_DCGAN_results/mnist'):
    os.mkdir('MNIST_DCGAN_results/mnist')
test_torch = mnist.read_image_file(os.path.join('data/raw/','t10k-images-idx3-ubyte'))
# test_torch.size()
tor2numpy=test_torch.numpy()
# tor2numpy.shape
# %matplotlib inline
# plt.figure(figsize=(64,64),dpi=1)
# plt.imshow(tor2numpy[0], cmap='gray')

for i in range(0,1000):
    dst=transform.resize(tor2numpy[i], (64, 64))
    misc.imsave('MNIST_DCGAN_results/mnist/'+str(i)+".jpg",dst)



# In[4]:




# network
G = generator(128)
small_size = 2
small_G = generator(small_size)
# G.weight_init(mean=0.0, std=0.02)
# small_G.weight_init(mean=0.0, std=0.02)
G.cuda()
small_G.cuda()


# In[5]:


if os.path.exists("MNIST_DCGAN_results/state.pkl"):
    checkpoint = torch.load("MNIST_DCGAN_results/state.pkl")
    G.load_state_dict(checkpoint['G'])
if os.path.exists("MNIST_DCGAN_results/state_small_"+ str(small_size)+ ".pkl"):
    checkpoint = torch.load("MNIST_DCGAN_results/state_small_"+ str(small_size)+ ".pkl")
    small_G.load_state_dict(checkpoint['small_G'])


# In[13]:


z_ = torch.randn((1000, 100)).view(-1, 100, 1, 1)
z_ = Variable(z_.cuda(), volatile=True)
G.eval()
small_G.eval()
test_images = G(z_)
test_images_small = small_G(z_)


# In[14]:



images_numpy = test_images.cpu().data.numpy()
# images_numpy.shape


# In[15]:


images_small_numpy = test_images_small.cpu().data.numpy()
# images_small_numpy.shape


# In[16]:


if not os.path.isdir('MNIST_DCGAN_results/G'):
    os.mkdir('MNIST_DCGAN_results/G')
if not os.path.isdir('MNIST_DCGAN_results/small_G_'+ str(small_size)):
    os.mkdir('MNIST_DCGAN_results/small_G_'+ str(small_size))

for i in range(0,1000):
    dst=transform.resize(images_numpy[i][0], (64, 64))
    dst_small = transform.resize(images_small_numpy[i][0], (64, 64))
    misc.imsave('MNIST_DCGAN_results/G/'+str(i)+".jpg",dst)
    misc.imsave('MNIST_DCGAN_results/small_G_'+ str(small_size)+'/'+str(i)+".jpg",dst_small)

