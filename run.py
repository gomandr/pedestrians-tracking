import argparse
import numpy as np

from PIL import Image, ImageDraw
from Video_to_Image import Video_to_Stack, Stack_to_Video
from bounding_boxes import predict_bboxes
from bbox_to_image import bbox_to_image
from delete_folder_content import delete_folder_content
from features import features
from utils import compute_center, dist

##############################################################################
### Options and parameters
parser = argparse.ArgumentParser(description='Pedestrian re-identification')
parser.add_argument('--path_to_video', default='./Videos/MOT16-10-raw.webm', type=str, help='path to the input video')
parser.add_argument('--path_to_stack', default='./images/new_frames/video1', type=str, help='path the output images stack')
parser.add_argument('--path_to_save', default='./Final_videos/final_video9.avi', type=str, help='path the output video')
parser.add_argument('--show', default=False, type=bool, help='show each frame after processing')

opt = parser.parse_args()

# Path to the input video 
#path_to_video = '.\Videos\MOT16-10-raw.webm' # == video1
#path_to_video = '.\Videos\MOT16-11-raw.webm' # == video2
#path_to_video = '.\Videos\MOT16-13-raw.webm' # == video3
path_to_video = opt.path_to_video

# Path for the output video
path_to_save = opt.path_to_save
path_to_stack = opt.path_to_stack
show = opt.show

# Thresholds
score_min_bbox = 0.8
score_min_features = 0.7
thresh_centers = 20


##############################################################################
### Get the frames from the input video
frames = Video_to_Stack(path_to_video)

##############################################################################
### Main
#bboxes = predict_bboxes(frames[:2], path_to_model='Models/resnet50_ped', score_min=0.5, show_pred=True)
#bboxes_query = bboxes[0]
#gallery = bboxes[1]
#frame = frames[0]

# Dictionnary with query informations
query_inf = dict()  # key=label --> [bbox, features, last frame]


for i, frame in enumerate(frames):

    frame = frames[i]
    print('#### Frame', i+1, '/', len(frames), '####', flush=True)
    print('Extracting bounding boxes from new frame...', flush=True)
    bboxes = predict_bboxes(frame, path_to_model='Models/resnet50_ped', \
                            score_min=score_min_bbox, show_pred=False)
    if i==0: # First frame
        print('Deleting old query images ...', flush=True)
        delete_folder_content(folder='./images/query/persons/')
        print('Saving query images ...', flush=True)
        bbox_to_image(frame, bboxes, path='./images/query/persons/') # All bboxes are saved as query
        
        # Get all images through the features extraction 
        print('Computing query features ...', flush=True)
        original_qf = np.array(features(images_dir = './images/query'))
        
        # Create label associated to each query features (# = index + 1)
        labels = (np.arange(len(original_qf))+1).astype(int)
        
        # Filling the dictionnary
        for j, label in enumerate(labels) :
            query_inf[label] = [bboxes[j], original_qf[j]]

            
    else: # Any other frame
        # Extract new bounding boxes as images and save them
        print('Deleting old gallery images ...', flush=True)
        delete_folder_content(folder='./images/gallery/persons/')
        
        print('Saving new gallery images ...', flush=True)
        bbox_to_image(frame, bboxes, path='./images/gallery/persons/')
        
        # Computes features of all the new bounding boxes == gallery
        print('Computing gallery features ...', flush=True)
        original_gf = np.array(features(images_dir = './images/gallery'))
        gf = original_gf.copy()
        new_qf = original_qf.copy()
    
        # Check where all query features are in the gallery 
        print('Comparing gallery features with query ones ...', flush=True)
        labels = np.zeros(len(original_gf)).astype(int)
        idx_origin = np.arange(0,len(original_gf))
        
        scores = np.dot(original_qf,original_gf.T)
        scores = np.append([np.arange(1,len(original_gf)+1)],scores, axis=0)
        scores = np.append([np.arange(0,len(original_qf)+1)],scores.T, axis=0) # len(gf) x len(qf)
        
        while min(scores.shape) > 1:
            score_max = np.amax(scores[1:,1:])
            
            # If the score is still high enough => assign as same person
            if score_max > score_min_features:
                idx_max = np.where(scores==score_max)
                idx_gf = scores[idx_max[0],0].astype(int)-1 # !! +1 relative to the true idx 
                idx_qf = scores[0,idx_max[1]].astype(int)-1 # !! +1 relative to the true idx 
                
                # If the boxes are close and approximatively the same scale it can be the same person
                center_q = compute_center(query_inf[idx_qf[0]+1][0])
                center_g = compute_center(bboxes[idx_gf[0]])
                dist_centers = dist(center_q, center_g)
                
                if dist_centers < thresh_centers :
                    
                    # Replace old query feature with the new one found in the gallery
                    new_qf[idx_qf] = gf[idx_gf]
                    
                    # Assign the gallery bounding box the right label (same as previously)
                    labels[idx_gf] = idx_qf+1
                    
                    # Dictionnary update
                    query_inf[idx_qf[0]+1] = [bboxes[idx_gf[0]], gf[idx_gf[0]]]
                    
                    # Take the query and gallery features associated out of the scores table
                    scores = np.delete(scores, idx_max[0], axis=0)
                    scores = np.delete(scores, idx_max[1], axis=1)
                
                # If not the high score is not justified and put to zeros
                else :
                    scores[idx_max] = 0
   
            else: # No more scores high enough
                break
            
        # Non-assigned gallery features => add them directly as new queries
        if scores.shape[0] > 1:
            idx_remaining_gf = scores[1:,0].astype(int)-1
            gf = gf[idx_remaining_gf,:]
            new_labels = np.arange(len(original_qf)+1,len(original_qf)+len(scores))
            labels[np.where(labels==0)] = new_labels
            new_qf = np.append(new_qf, gf, axis=0)
            
            # Add the new queries to the dictionnary
            for l, label in enumerate(new_labels) :
                query_inf[label] = [bboxes[idx_remaining_gf[l]], gf[l]]
                
        print('Assigning new query features ...', flush=True)
        original_qf = new_qf
            
    # Plot new frame with bounding boxes and labels
    if show==True:
        print('Preparing plot ...', flush=True)            
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        for k in range(len(bboxes)):    
            draw.rectangle(bboxes[k].tolist(), outline='red') 
            draw.text((bboxes[k][0],bboxes[k][1]-10), str(labels[k]),fill=(0,255,0,255))
        img.save(path_to_stack+'\Frame'+str(i)+'.png','png')
        img.show()

# Creation of the video
Stack_to_Video(path_to_stack, path_to_save)