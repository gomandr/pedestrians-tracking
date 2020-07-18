from PIL import Image, ImageDraw
import torch
from get_resnet50_fpn import get_resnet50_fpn
from get_detection_model import get_detection_model
from ECPdataset import ECPdataset
from engine import get_transform
from torchvision.transforms import functional as F
import argparse

"""
To run in the console, input commands:
    
python run_predict.py -i '../ECP/day/img/train/berlin/berlin_00000.png' 
              -m 'saves/resnet50_amst_8bst'
              
with: 
    
-i the path from where to load the image to make predictions on
-m the path from where to load the pretrained model to use for prediction

"""

def predict_bboxes(path_to_image = '../ECP/day/img/train/berlin/berlin_00000.png', \
                   path_to_model = 'saves/resnet50_amst_8bst'):
    
    if path_to_image == None:
        path_to_image = '../ECP/day/img/train/berlin/berlin_00000.png'
    
    if path_to_model == None:
        path_to_model = 'saves/resnet50_amst_8bst'
        
    # Three classes - background, pedestrian, rider
    num_classes = 3
    model = get_resnet50_fpn(num_classes)
    #model = get_detection_model(num_classes, model='mobilenet', anch='15-highersize')
    
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    image_pred = Image.open(path_to_image).convert("RGB")
    image_pred = F.to_tensor(image_pred)
    
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model.eval()
    with torch.no_grad():
        prediction = model([image_pred])
        
    print('Position of bounding-boxes:\n', prediction[0]['boxes'])
    print('Labels of bounding-boxes (1==pedestrian, 2==rider):\n', prediction[0]['labels'])
            
    img = Image.fromarray(image_pred.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(img)
    for i in range(len(prediction[0]['boxes'])):
        if prediction[0]['labels'][i] == 1:
            color = 'red'
        elif prediction[0]['labels'][i] == 2:
            color = 'blue'
        draw.rectangle(prediction[0]['boxes'][i].cpu().numpy(), outline = color) 
    img.show()
    return prediction

def main(path_to_image, path_to_model):
    prediction = predict_bboxes(path_to_image, path_to_model)
    labels = prediction[0]['labels']
    pos = prediction[0]['boxes']

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--path_image", required=False, help="path to test image")
    ap.add_argument("-m", "--path_model", required=False, help="path to model")

    args = vars(ap.parse_args())
    main(args["path_image"],args["path_model"])
    