"""
Copyright (C) 2018 Cuiyirui.  All rights reserved.
"""
from models.models import create_model
from models.utils import get_test_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images,Timer,to_1_channel
import html
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
from util.visualizer import save_images
from util import html
import numpy as np
import sys
#import tensorboardX
import shutil
torch.cuda.set_device(1)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,default='./configs/edges2shirts_stripe_patch_folder.yaml', help="net configuration")
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--E_path', type=str, default='./pretrained_models/latest_net_E.pt', help="checkpoint of autoencoders")
parser.add_argument('--G_path', type=str, default='./pretrained_models/latest_net_G.pt', help="checkpoint of generator")
parser.add_argument('--D_path', type=str, default='./pretrained_models/latest_net_D.pt', help="checkpoint of discriminator")
parser.add_argument("--continue_train",action="store_true")
parser.add_argument('--phase',type=str,default='test_demo_stripe_patch',help='model phase')
parser.add_argument('--input', type=str, default='./inputs/contour2shirt', help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--batchsize', type=int, default=1, help="batch size when testing")
parser.add_argument('--num_workers', type=int, default=8, help="num of workers")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--trainer', type=str, default='myMUNIT_patch', help="MUNIT|UNIT|myMUNIT|myMUNIT_patch|myNet")

opts = parser.parse_args()

def to_img(outputs):
    output_im = outputs[0].data.cpu().numpy()
    output_im= ((output_im + 1) / 2.)*255
    output_im = output_im.astype('uint8')
    output_im = np.transpose(output_im, (1, 2, 0))
    return output_im

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path
# creat model
trainer = create_model(opts,config)
trainer.cuda()
# creat data loader
test_loader_a, test_loader_b = get_test_data_loaders(opts)
# whether continue train
trainer.load_model_dict(opts)
# create website
web_dir = os.path.join(opts.output_folder, opts.phase +
                       '_sync' if opts.synchronized else opts.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, G = %s, E = %s' % (
    'contour2shirt', opts.phase, opts.G_path, opts.E_path))

# start test
for it, (images_a, images_b) in enumerate(zip(test_loader_a, test_loader_b)):
    trainer.update_learning_rate()
    images_a, images_b = images_a.cuda(), images_b.cuda()
    trainer.set_input(images_a, images_b, config)
    encoded_output, _ = trainer.test_encoded()
    # add input domain A (contour) image to list
    input = images_a[:, :, :, 0:256]
    all_images = [to_img(input)]
    all_names = ['input']
    # add input domain B (ground truth) image to list
    ground = images_b[:, :, :, 0:256]
    all_images.append(to_img(ground))
    all_names.append('ground_truth')
    # add encoded image to list
    encoded_output_im = to_img(encoded_output)
    path = os.path.join(opts.output_folder, 'encoded_output.jpg')
    all_images.append(encoded_output_im)
    all_names.append('encoded_output.jpg')
    # add random encoded image to list
    for j in range(opts.num_style):
        # to image
        random_output, _ = trainer.test_sample()
        output_im = to_img(random_output)
        path = os.path.join(opts.output_folder, 'output{:03d}.jpg'.format(j))
        all_images.append(output_im)
        all_names.append('output{:03d}.jpg'.format(j))
        # vutils.save_image(outputs.data, path, padding=0, normalize=True)
    else:
        pass
    print("process test image_%s ..." % str(it + 1))
    # save web
    save_images(webpage, all_images, all_names, 'sample{0:3d}'.format(it + 1), None,
                width=256, aspect_ratio=1.0)

webpage.save()