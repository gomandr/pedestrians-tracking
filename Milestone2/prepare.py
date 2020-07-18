import os
from shutil import copy2
import pandas as pd

# You only need to change this line to your dataset download path
dataset_path = '../../../scratch/adaimi/msmt17/MSMT17_V1'
save_path = '../../../scratch/angomes/Milestone2'

if not os.path.isdir(dataset_path):
    print('please change the download_path')

                ### Training ###


if not os.path.isdir(save_path+'/split_train'):
    os.mkdir(os.path.join(save_path,'split_train'))
 

#-----------------------------------------
#train
train_path = dataset_path + '/train'
train_save_path = save_path+'/split_train/train'
if not os.path.isdir(train_save_path):
    os.mkdir(os.path.join(train_save_path))

data_train = pd.read_csv(dataset_path+'/list_train.txt', header = None, sep=' ')


for name in data_train[0] :
    ID = name.split('/')[0]
    if not os.path.isdir(train_save_path+'/'+ID):
        os.mkdir(os.path.join(train_save_path+'/'+ID))
    if not name[-3:]=='jpg':
        continue
    src_path = os.path.join(train_path, name)
    copy2(src_path, os.path.join(train_save_path,ID))


#-----------------------------------------
#valid
val_path = dataset_path + '/train'
val_save_path = save_path + '/split_train/val'
if not os.path.isdir(val_save_path):
    os.mkdir(os.path.join(val_save_path))

data_val = pd.read_csv(dataset_path+'/list_val.txt', header = None, sep=' ')

for name in data_val[0] :
    ID = name.split('/')[0]
    if not os.path.isdir(val_save_path+'/'+ID):
        os.mkdir(os.path.join(val_save_path+'/'+ID))
    if not name[-3:]=='jpg':
        continue
    src_path = os.path.join(val_path, name)
    copy2(src_path, os.path.join(val_save_path,ID))


                ### Testing ###

if not os.path.isdir(save_path+'/split_test'):
    os.mkdir(os.path.join(save_path,'split_test'))
    

#-----------------------------------------
#query
query_path = dataset_path + '/test'
query_save_path = save_path + '/split_test/query'
if not os.path.isdir(query_save_path):
    os.mkdir(os.path.join(query_save_path))

data_query = pd.read_csv(dataset_path+'/list_query.txt', header = None, sep=' ')

for name in data_query[0] :
    ID = name.split('/')[0]
    if not os.path.isdir(query_save_path+'/'+ID):
        os.mkdir(os.path.join(query_save_path+'/'+ID))
    if not name[-3:]=='jpg':
        continue
    src_path = os.path.join(query_path, name)
    copy2(src_path, os.path.join(query_save_path, ID))


#-----------------------------------------
#gallery
gallery_path = dataset_path + '/test'
gallery_save_path = save_path + '/split_test/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(os.path.join(gallery_save_path))

data_gallery = pd.read_csv(dataset_path+'/list_gallery.txt', header = None, sep=' ')

for name in data_gallery[0] :
    ID = name.split('/')[0]
    if not os.path.isdir(gallery_save_path+'/'+ID):
        os.mkdir(os.path.join(gallery_save_path+'/'+ID))
    if not name[-3:]=='jpg':
        continue
    src_path = os.path.join(gallery_path, name)
    copy2(src_path, os.path.join(gallery_save_path, ID))