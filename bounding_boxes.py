from PIL import Image, ImageDraw
import torch
from get_resnet50_fpn import get_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np

def predict_bboxes(frames, path_to_model='Models/resnet50_ped', score_min=0, show_pred=False):
            
    num_classes = 2
    model = get_resnet50_fpn(num_classes)
    
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    checkpoint = torch.load(path_to_model, map_location='cpu') # map_location='cuda:0', map_location='cpu' 
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    with torch.no_grad():
        
        predictions = []
        #for i, frame in enumerate(frames):
        #print('Predicting bboxes of frame', i+1, '/', str(len(frames)))
        frame = frames
        image_pred = F.to_tensor(frame)
        prediction = model([image_pred]) 
        
        idx_to_keep = np.where(prediction[0]['scores'] > score_min)
        
        bboxes = prediction[0]['boxes'][idx_to_keep]
        bboxes = np.array(bboxes).astype(int)
        
        if show_pred == True:                
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            for i in range(len(bboxes)):
                draw.rectangle(bboxes[i].tolist(), outline='red') 
            img.show()
                
        
            #predictions.append(bboxes)             
    
    return bboxes 

    
