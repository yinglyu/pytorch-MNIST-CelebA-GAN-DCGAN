from compress_iter import generator
from thop import profile
from thop import clever_format
import torch
from torch.autograd import Variable
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('size', type=int, default=128,
                    help=('Size of generator'))
def param(size = 128):
    G = generator(size)
    G.weight_init(mean=0.0, std=0.02)
    G.cuda()
    path = "MNIST_DCGAN_results/iter/best_state_"+ str(size)+ ".pkl"
    checkpoint = torch.load("MNIST_DCGAN_results/iter/best_state_"+ str(size)+ ".pkl")
    G.load_state_dict(checkpoint['G'])
    z_ = torch.randn((1, 100)).view(-1, 100, 1, 1)
    
    z_ = Variable(z_.cuda(), volatile=True)
    flops, params = profile(G, inputs=(z_, ))
    flops, params = clever_format([flops, params], "%.3f")
    print ("flops="+str(flops))
    print ("params="+str(params))
    return flops, params

if __name__ == '__main__':
    args = parser.parse_args()
    param(args.size)

