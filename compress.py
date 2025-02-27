import os, time, sys
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import fid_score
import skimage
import cv2
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.filterwarnings("ignore")

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

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise
with torch.no_grad():
    fixed_z_ = Variable(fixed_z_.cuda())
def show_result(num_epoch, show = False, save = False, path = 'result', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)
    
    G.eval()
    small_G.eval()
    if isFix:
        test_images = G(fixed_z_)
        test_images_small = small_G(fixed_z_)
    else:
        test_images = G(z_)
        test_images_small = small_G(z_)
    G.train()
    small_G.train()
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+"_big.png")

    if show:
        plt.show()
    else:
        plt.close()
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images_small[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+"_small.png")

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['G_losses']))


 #   y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
 #   plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('size', type=int, default=32,
                    help=('Size of small generator')) 
args = parser.parse_args()

# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20

# data_loader
img_size = 64
# transform = transforms.Compose([
#         transforms.Scale(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])

transform = transforms.Compose([
transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G = generator(128)
small_size = args.size
small_G = generator(small_size)
# D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
small_G.weight_init(mean=0.0, std=0.02)
# D.weight_init(mean=0.0, std=0.02)
G.cuda()
small_G.cuda()
# D.cuda()

# Binary Cross Entropy loss
# BCE_loss = nn.BCELoss()
L1_loss = nn.L1Loss()

# Adam optimizer
G_optimizer = optim.Adam(small_G.parameters(), lr=lr, betas=(0.5, 0.999))
# D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('MNIST_DCGAN_results'):
    os.mkdir('MNIST_DCGAN_results')
if not os.path.isdir('MNIST_DCGAN_results/Random_results'):
    os.mkdir('MNIST_DCGAN_results/Random_results')
if not os.path.isdir('MNIST_DCGAN_results/Fixed_results'):
    os.mkdir('MNIST_DCGAN_results/Fixed_results')
if not os.path.isdir('MNIST_DCGAN_results/Compress/'):
    os.mkdir('MNIST_DCGAN_results/Compress/')
if not os.path.isdir('MNIST_DCGAN_results/Compress/'+ str(small_size)):
    os.mkdir('MNIST_DCGAN_results/Compress/'+ str(small_size))
if not os.path.isdir('MNIST_DCGAN_results/Compress/'+ str(small_size)+ '/Training'):
    os.mkdir('MNIST_DCGAN_results/Compress/'+ str(small_size)+ '/Training')
if not os.path.isdir('MNIST_DCGAN_results/Compress/'+ str(small_size)+ '/Validation'):
    os.mkdir('MNIST_DCGAN_results/Compress/'+ str(small_size)+ '/Validation')
if not os.path.isdir('MNIST_DCGAN_results/FID/'):
    os.mkdir('MNIST_DCGAN_results/FID/')
    
if os.path.exists("MNIST_DCGAN_results/state.pkl"):
    checkpoint = torch.load("MNIST_DCGAN_results/state.pkl")
    G.load_state_dict(checkpoint['G'])
if os.path.exists("MNIST_DCGAN_results/FID/state_small_"+ str(small_size)+ ".pkl"):
    checkpoint = torch.load("MNIST_DCGAN_results/FID/state_small_"+ str(small_size)+ ".pkl")
    small_G.load_state_dict(checkpoint['small_G'])
#     D.load_state_dict(checkpoint['D'])
    G_optimizer.load_state_dict(checkpoint['G_optimizer'])
#     D_optimizer.load_state_dict(checkpoint['D_optimizer'])
    train_hist = checkpoint['train_hist']
    total_ptime =  checkpoint['total_ptime']
    start_epoch = checkpoint['epoch']
    best_FID = checkpoint['best_FID']
    print("start from epoch" + str(start_epoch))
    #num_iter = start_epoch
    
else:
    start_epoch = 0
    total_ptime = 0
    train_hist = {}
    best_FID = sys.maxsize
    #train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    #num_iter = start_epoch 

print('training start!')
# start_time = time.time()
for epoch in range(start_epoch, train_epoch):
#     D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    num_iter = 0
    for x_, _ in train_loader:
        # train discriminator D
        # D.zero_grad()
        mini_batch = x_.size()[0]
#         y_real_ = torch.ones(mini_batch)
#         y_fake_ = torch.zeros(mini_batch)
#         x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
#         D_result = D(x_).squeeze()
#         D_real_loss = BCE_loss(D_result, y_real_)
#         z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
#         z_ = Variable(z_.cuda())
#         G_result = G(z_)
        # D_result = D(G_result).squeeze()
        # D_fake_loss = BCE_loss(D_result, y_fake_)
        # D_fake_score = D_result.data.mean()
        # D_train_loss = D_real_loss + D_fake_loss
        # D_train_loss.backward()
        # D_optimizer.step()
        # D_losses.append(D_train_loss.data[0])
        # D_losses.append(D_train_loss.data[0])
        # train generator G
        G.zero_grad()
        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())

        # G_result = G(z_).detach()
        # small_G_result = small_G(z_)
        # D_result = D(G_result).squeeze()
        # G_train_loss = BCE_loss(D_result, y_real_)
        # G_train_loss = L1_loss(G_result, small_G_result)
        G_result = G(z_)
        small_G_result = small_G(z_)
        G_train_loss = L1_loss( small_G_result, G_result.detach())
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.item())
        #G_losses.append(G_train_loss.data[0])

        num_iter += 1
#         if ((num_iter+1)%100 == 0):
#             p = 'MNIST_DCGAN_results/Compress/'+ str(small_size)+ '/Training/MNIST_DCGAN_epoch_' + str(epoch + 1) + '_iter_' + str(num_iter+1) 
#             show_result((epoch+1), save=True, path=p, isFix=False)
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

    print('[%d/%d] - ptime: %.2f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, 
                                                              torch.mean(torch.FloatTensor(G_losses))))
    #p = 'MNIST_DCGAN_results/Compress/Random_results/MNIST_DCGAN_' + str(epoch + 1) 
    fixed_p = 'MNIST_DCGAN_results/Compress/'+ str(small_size)+ '/Validation/MNIST_DCGAN_' + str(epoch + 1) 
    #show_result((epoch+1), save=True, path=p, isFix=False)
    #show_result((epoch+1), save=True, path=fixed_p, isFix=True)
    #train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    total_ptime += per_epoch_ptime
    
    #export jpg
    z_ = torch.randn((1000, 100)).view(-1, 100, 1, 1)
    with torch.no_grad():
        z_ = Variable(z_.cuda())
    test_images_small = small_G(z_)
    images_small_numpy = test_images_small.cpu().data.numpy()
    torch.cuda.empty_cache()
    
    p_small_G = "MNIST_DCGAN_results/RGB/small_G_" + str(small_size)
    if not os.path.isdir(p_small_G):
        os.mkdir(p_small_G)
    p_mnist = "MNIST_DCGAN_results/RGB/mnist"
    if os.path.exists(p_mnist+".npz"):
        p_mnist = p_mnist+".npz"
    #path = [p_mnist, p_small_G]
    for i in range(0,1000):
        src=skimage.transform.resize(images_small_numpy[i][0], (64, 64))
        imageio.imwrite("temp.jpg",skimage.img_as_ubyte(src))
        src = cv2.imread("temp.jpg", 0)
        src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(p_small_G + '/'+str(i)+".jpg", src_RGB)
    
    fid = fid_score.calculate_fid_given_paths([p_mnist, p_small_G], 50, True,  2048)
    if fid < best_FID:
        best_FID = fid
        print("best_FID:" + str(fid))
        state = {'small_G': small_G.state_dict(), 'G_optimizer': G_optimizer.state_dict(),'train_hist' : train_hist, 'epoch': epoch+1, 'total_ptime' : total_ptime, 'best_FID' : best_FID}
        torch.save(state, "MNIST_DCGAN_results/FID/state_small_"+ str(small_size)+ ".pkl")
    if fid < best_FID:
        best_FID = fid
        print("best_FID:" + str(fid))
        state = {'small_G': small_G.state_dict(), 'G_optimizer': G_optimizer.state_dict(),'train_hist' : train_hist, 'epoch': epoch+1, 'total_ptime' : total_ptime, 'best_FID' : best_FID}
        torch.save(state, "MNIST_DCGAN_results/FID/best_state_small_"+ str(small_size)+ ".pkl")
    state = {'small_G': small_G.state_dict(), 'G_optimizer': G_optimizer.state_dict(),'train_hist' : train_hist, 'epoch': epoch+1, 'total_ptime' : total_ptime, 'best_FID' : best_FID}
    torch.save(state, "MNIST_DCGAN_results/FID/state_small_"+ str(small_size)+ ".pkl")
    
# end_time = time.time()
# total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(small_G.state_dict(), "MNIST_DCGAN_results/Compress/"+ str(small_size)+ "/generator_param.pkl")
# torch.save(D.state_dict(), "MNIST_DCGAN_results/discriminator_param.pkl")
with open('MNIST_DCGAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

#show_train_hist(train_hist, save=True, path='MNIST_DCGAN_results/MNIST_DCGAN_train_hist.png')

# images = []
#for e in range(train_epoch):
#    img_name = 'MNIST_DCGAN_results/Fixed_results/MNIST_DCGAN_' + str(e + 1) + '.png'
#    images.append(imageio.imread(img_name))
#imageio.misave('MNIST_DCGAN_results/generation_animation.gif', images, fps=5)