"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder
from trainer import MUNIT_Trainer, UNIT_Trainer,to_image
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from util.visualizer import save_images
from util import html


parser = argparse.ArgumentParser()
parser.add_argument('--phase',type=str,default='test_demo_stripe3',help='model phase')
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, default='./inputs/contour2shirt', help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--G_path', type=str, help="checkpoint of autoencoders")
parser.add_argument('--E_path', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")


opts = parser.parse_args()
# translate to im
def to_img(outputs):
    output_im = outputs[0].data.cpu().numpy()
    output_im= ((output_im + 1) / 2.)*255
    output_im = output_im.astype('uint8')
    output_im = np.transpose(output_im, (1, 2, 0))
    return output_im


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)


# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")
state_dict_E = torch.load(opts.E_path)
state_dict_G = torch.load(opts.G_path)
# load net by path
trainer.enc_a.load_state_dict(state_dict_E['a'])
trainer.enc_b.load_state_dict(state_dict_E['b'])
trainer.gen_a.load_state_dict(state_dict_G['a'])
trainer.gen_b.load_state_dict(state_dict_G['b'])

trainer.cuda()
trainer.eval()
encode = trainer.enc_a if opts.a2b else trainer.enc_b # encode function
style_encode = trainer.gen_b if opts.a2b else trainer.gen_a # encode function
decode = trainer.gen_b if opts.a2b else trainer.gen_a # decode function

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b==1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

transform = transforms.Compose([transforms.Resize(new_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# create website
web_dir = os.path.join(opts.output_folder, opts.phase +
                       '_sync' if opts.synchronized else opts.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, G = %s, E = %s' % (
    'contour2shirt', opts.phase, opts.G_path, opts.E_path))

# test image
image_path = opts.input + '/'+ opts.phase+'/contour/'
num_image = os.listdir(image_path)
for i in range(len(num_image)):
    contour_name = opts.input + '/'+ opts.phase+ '/contour/' + str(i+1) + '.png'
    ground_name = opts.input + '/'+ opts.phase + '/ground/' + str(i+1) + '.png'
    image = Variable(transform(Image.open(contour_name).convert('RGB')).unsqueeze(0).cuda(), volatile=True)
    style_image = Variable(transform(Image.open(ground_name).convert('RGB')).unsqueeze(0).cuda(), volatile=True)
    # Start testing
    image_code = encode(image)

    if opts.trainer == 'MUNIT':
        rand_style= Variable(torch.randn(opts.num_style,style_dim).cuda(), volatile=True)
        encoded_style = encode(style_image)
        style = rand_style

        # add input image to list
        input=image
        all_images=[to_img(input)]
        all_names=['input']

        # add ground truth to list
        ground = style_image
        all_images.append(to_img(ground))
        all_names.append('ground_truth')

        # add encoded image to list
        encoded_s = encoded_style
        encoded_output = decode(image,encoded_s)
        encoded_output_im = to_img(encoded_output)
        path = os.path.join(opts.output_folder, 'encoded_output.jpg')
        all_images.append(encoded_output_im)
        all_names.append('encoded_output.jpg')
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(image, s)
            # to image

            output_im = to_img(outputs)
            path = os.path.join(opts.output_folder, 'output{:03d}.jpg'.format(j))

            all_images.append(output_im)
            all_names.append('output{:03d}.jpg'.format(j))
            #vutils.save_image(outputs.data, path, padding=0, normalize=True)
    elif opts.trainer == 'UNIT':
        outputs = decode(image)
        outputs = (outputs + 1) / 2.
        path = os.path.join(opts.output_folder, 'output.jpg')
        vutils.save_image(outputs.data, path, padding=0, normalize=True)
    else:
        pass
    print("process test image_%s ..." % str(i + 1))
    # save web
    save_images(webpage, all_images, all_names, 'sample{0:3d}'.format(i+1), None,
                    width=256, aspect_ratio=1.0)



webpage.save()

#if not opts.output_only:
    # also save input images
    # vutils.save_image(image.data, os.path.join(opts.output_folder, 'input.jpg'), padding=0, normalize=True)



