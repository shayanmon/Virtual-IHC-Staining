import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
import math
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
# opt = TrainOptions().parse()
opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
# iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
# if opt.continue_train:
#     try:
#         start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
#     except:
#         start_epoch, epoch_iter = 1, 0
#     print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
# else:    
#     start_epoch, epoch_iter = 1, 0

# opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
# if opt.debug:
#     opt.display_freq = 1
#     opt.print_freq = 1
#     opt.niter = 1
#     opt.niter_decay = 0
#     opt.max_dataset_size = 10

# data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()
# dataset_size = len(data_loader)
# print('#training images = %d' % dataset_size)
print(opt.isTrain)
opt.isTrain = False
print(opt.isTrain)
model = create_model(opt)
visualizer = Visualizer(opt)


## Test to load data in the inference step
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
    # print('aca llego')
    generated = model.inference(data['label'], data['inst'], data['image'])
    
    
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    # print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

# Correr este codigo: 
# python train_test.py --dataroot /group/anantm-g00/jsalgu2/datasets/HE2IHC_data --name HE2IHC --which_epoch 60 --batchSize 16 --how_many 5773 --gpu_ids 0,1 --label_nc 0 --no_instance --resize_or_crop none --results_dir /scratch/jsalgu2/datasets/results/


# python train_test.py --dataroot /home/jsalgu2/nfs/HE_to_IHC_code/pix2pixHD/datasets/HE2IHC_data --name HE2IHC --which_epoch 60 --batchSize 16 --how_many 5773 --gpu_ids 0,1 --label_nc 0 --no_instance --resize_or_crop none --results_dir /home/jsalgu2/nfs/HE_to_IHC_code/Inference_vanderbilt/Allinferencepatches_77

