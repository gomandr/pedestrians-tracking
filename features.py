from __future__ import print_function, division
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
import os
import yaml
from model_reid import PCB, PCB_test

def features(images_dir='images/query', save_dir='models/PCB', \
             gpu_ids=[], batch_size=32, num_workers=0):

    ###load config###
    # load the training config
    config_path = os.path.join(save_dir,'opts.yaml')
    
    with open(config_path, 'r') as stream:
            config = yaml.load(stream)
    nclasses = config['nclasses']
    
    # set gpu ids
    if len(gpu_ids)>0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    
    ######################################################################
    # Load Data
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    
    image_datasets = datasets.ImageFolder(images_dir ,data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    ######################################################################
    # Load collected data and trained model
    model_structure = PCB(nclasses)
    model = load_network(model_structure, save_dir)
    model = PCB_test(model)
    
    # Change to test mode
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        cuda=True
    else:
        cuda=False
    
    # Extract feature
    with torch.no_grad():
        feature = extract_feature(model, dataloaders, cuda)
    
    return feature

def load_network(network, save_dir):
    save_path = os.path.join(save_dir,'net.pth')
    network.load_state_dict(torch.load(save_path))
    return network

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders,cuda):
    features = torch.FloatTensor()
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        if cuda == True:
            ff = torch.FloatTensor(n,512).zero_().cuda()
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts
        else:
            ff = torch.FloatTensor(n,512).zero_()
            ff = torch.FloatTensor(n,2048,6).zero_()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            if cuda == True:    
                input_img = Variable(img.cuda())
            else:
                input_img = Variable(img)
            outputs = model(input_img) 
            ff += outputs
        # norm feature
        # feature size (n,2048,6)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)

        features = torch.cat((features,ff.data.cpu()), 0)
    return features