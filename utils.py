import numpy as np

def compute_center(box) :
    center = np.zeros(2)
    center[0] = int(abs(box[0]-box[2])/2)
    center[1] = int(abs(box[1]-box[3])/2)
    
    return center

def dist(point1, point2) :
    dist = ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**(1/2)
    
    return dist