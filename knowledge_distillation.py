from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import compress_iter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--sizes', type=int, nargs='+', 
                    help=('Sizes of iterative generator')) 

args = parser.parse_args()
for i in range (0, len(args.sizes)-1):
    print(str(args.sizes[i]) + " to " + str(args.sizes[i+1]))
    compress_iter.compress(args.sizes[i], args.sizes[i+1])
    