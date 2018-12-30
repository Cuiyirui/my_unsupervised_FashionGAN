"""
Copyright (C) 2018 Cuiyirui.  All rights reserved.
"""
from models.models import create_model
from models.utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images,Timer,to_1_channel
from util.visualizer import Visualizer
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
import sys
#import tensorboardX
import shutil
torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/edges2single_shirts_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--E_path', type=str, default='./pretrained_models/latest_net_E.pt', help="checkpoint of autoencoders")
parser.add_argument('--G_path', type=str, default='./pretrained_models/latest_net_G.pt', help="checkpoint of generator")
parser.add_argument('--D_path', type=str, default='./pretrained_models/latest_net_D.pt', help="checkpoint of discriminator")
parser.add_argument("--continue_train",action="store_true")
parser.add_argument('--trainer', type=str, default='myNet', help="MUNIT|UNIT|myMUNIT|myMUNIT_patch|myMUNIT_within_patch|myVAE_MUNIT_patch|myNet")
parser.add_argument('--phase',type=str,default='train',help='which phase')
opts = parser.parse_args()

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
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
#train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
#train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()
# whether continue train
if opts.continue_train:
    trainer.load_model_dict(opts)



# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
#train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder



# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
visualizer = Visualizer(config['visdom_port'])
while True:
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda(), images_b.cuda()
        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.set_input(images_a, images_b,config)
            trainer.dis_update(config)
            trainer.gen_update(config)
            torch.cuda.synchronize()

        # visual result
        visualizer.display_current_results(trainer.get_current_visuals(), 1, ncols=2, save_result=False)
        save_result = False

        # visual error
        errors = trainer.get_current_errors(config)
        visualizer.print_current_errors(iterations//986+1, iterations, errors, 1)
        visualizer.plot_current_errors(iterations//986+1, iterations, errors)


        '''
        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            #write_loss(iterations, trainer, train_writer)
        '''
        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b,config)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        '''
        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')
        '''
        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1


        if iterations >= max_iter:
            sys.exit('Finish training')

