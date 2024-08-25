"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from climatetranslation.unit.trainer import UNIT_Trainer
from climatetranslation.unit.utils import prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, write_2images_data
from climatetranslation.unit.data import get_all_data_loaders

import argparse
import os
import sys
import shutil

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import tensorboardX


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/v7_example.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='/home/dfulu/model_outputs', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']

# data loaders
train_loader_a, test_loader_a, train_loader_b, test_loader_b = get_all_data_loaders(config)

# land masks
land_mask_a = train_loader_a.dataset.land_mask
land_mask_b = train_loader_b.dataset.land_mask

# Selection of climate fields to display after a number of updates
def generate_n(generator, n):
    return torch.cat([img for _, img in zip(range((n-1)//generator.batch_size + 1), generator)])[:n]

train_display_images_a = generate_n(train_loader_a, display_size).cuda()
train_display_images_b = generate_n(train_loader_b, display_size).cuda()
test_display_images_a  = generate_n(test_loader_a, display_size).cuda()
test_display_images_b  = generate_n(test_loader_b, display_size).cuda()


# Add some extra hyperpaameters with inferred info from data
hyperparams = config
hyperparams['input_dim_a'] = train_loader_a.dataset.shape[1]
hyperparams['input_dim_b'] = train_loader_b.dataset.shape[1]
hyperparams['land_mask_a'] = land_mask_a
hyperparams['land_mask_b'] = land_mask_b
print(hyperparams['input_dim_a'])


# Setup model and data loader
trainer = UNIT_Trainer(hyperparams)
trainer.cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# A small amount of datetimes have all NaN data
def all_nan_last_two_axis_any_channel(x):
    #return torch.any(torch.all(torch.all(torch.isnan(x), axis=-1), axis=-1), axis=-1)
    return torch.isnan(x).all(dim=-1).all(dim=-1).any()

def any_nan(x):
    #return torch.any(torch.all(torch.all(torch.isnan(x), axis=-1), axis=-1), axis=-1)
    return torch.isnan(x).any()

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0

while True:
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        # Skip NaN fields
        if any_nan(images_a) or any_nan(images_b):
            print('NaN detected. Skipping iteration = {}'.format(it))
            continue

        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if ((iterations + 1) % config['log_iter'] == 0) or (iterations==0):
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if ((iterations + 1) % config['image_save_iter'] == 0) or (iterations==0):
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                

            # only pass first 3 channels for image
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            write_2images_data(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images_data(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if ((iterations + 1) % config['image_display_iter'] == 0) or (iterations==0):
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')
            write_2images_data(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

