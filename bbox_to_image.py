from PIL import Image
import os

def bbox_to_image(frame, bboxes_query, path):
    for j, bbox_img in enumerate(bboxes_query): # Save each detected person as query
        q_img = frame[bbox_img[1]:bbox_img[3],bbox_img[0]:bbox_img[2]]
        im = Image.fromarray(q_img)
        if j < 10:
            im.save(os.path.join(path+'00'+str(j)+'.png'))
        elif j < 100:
            im.save(os.path.join(path+'0'+str(j)+'.png'))
        else:
            im.save(os.path.join(path+str(j)+'.png'))