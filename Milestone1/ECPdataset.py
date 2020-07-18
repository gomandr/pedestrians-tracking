import json
import os
import torch
import torch.utils.data
from PIL import Image

class ECPdataset(torch.utils.data.Dataset):
    def __init__(self, root, purp, transforms=None):
        self.root = root
        self.purp = purp
        self.transforms = transforms
        
        # load all image files, sorting them to ensure that they are aligned
#        self.imgs = list(sorted(os.listdir(os.path.join(root, "img/" + str(purp) ))))
#        self.labels = list(sorted(os.listdir(os.path.join(root, "label/" + str(purp) ))))
             
        self.imgs=list()
        for rt, _, files in os.walk(os.path.join(root, "img/" + str(purp)), topdown=False):
           for name in files:
              self.imgs.append(os.path.join(rt, name))
        self.imgs=sorted(self.imgs)
        
        self.labels=list()
        for rt, _, files in os.walk(os.path.join(root, "labels/" + str(purp)), topdown=False):
           for name in files:
              self.labels.append(os.path.join(rt, name))
        self.labels=sorted(self.labels)


    def __getitem__(self, idx):
        # load images and data 
        img_path = os.path.join(self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        data_path = os.path.join(self.labels[idx])
        with open(data_path) as json_file:
          data = json.load(json_file)

        # get boxes and identities
        boxes = []
        identities = []

        for i in range(len(data['children'])):
          identity = data['children'][i]['identity']
          if identity == 'pedestrian' or identity == 'rider':
            box = [data['children'][i]['x0'], data['children'][i]['y0'], \
                   data['children'][i]['x1'], data['children'][i]['y1']]
            boxes.append(box)
            if identity == 'pedestrian':
              identities.append(1)
            else:
              identities.append(2)

          for j in range(len(data['children'][i]['children'])):
            identity = data['children'][i]['children'][j]['identity']
            if identity == 'pedestrian' or identity == 'rider':
              box = [data['children'][i]['children'][j]['x0'], data['children'][i]['children'][j]['y0'], \
                     data['children'][i]['children'][j]['x1'], data['children'][i]['children'][j]['y1']]
              boxes.append(box)
              if identity == 'pedestrian':
                identities.append(1)
              else:
                identities.append(2)

        # to avoid error if no pedestrians or riders on image
        if len(boxes) == 0:
          boxes = [[0,0,1,1]]
          identities = [0]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        identities = torch.as_tensor(identities, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = identities
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)