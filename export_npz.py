

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


# network
size = 128
G = generator(size)
G.cuda()


if os.path.exists("MNIST_DCGAN_results/iter/best_state_"+ str(size) +".pkl"):
    checkpoint = torch.load("MNIST_DCGAN_results/iter/best_state_"+ str(size) +".pkl")
    G.load_state_dict(checkpoint['G'])


z_ = torch.randn((1000, 100)).view(-1, 100, 1, 1)
z_ = Variable(z_.cuda(), volatile=True)
G.eval()
test_images = G(z_)

images_numpy = test_images.cpu().data.numpy()

path_G = 'MNIST_DCGAN_results/RGB/G_'+ str(size)
if not os.path.isdir(path_G):
    os.mkdir(path_G)

for i in range(0,1000):
    src=skimage.transform.resize(images_numpy[i][0], (64, 64))
    imageio.imwrite("temp.jpg",skimage.img_as_ubyte(src))
    src = cv2.imread("temp.jpg", 0)
    src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(path_G +str(i)+".jpg", src_RGB)
