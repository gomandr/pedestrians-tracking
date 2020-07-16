import os
import cv2

from Video_to_Image import Stack_to_Video


Stack = []
stack_path = './images/new_frames/video1/test1/'
Video_path = './output.avi'
test = os.listdir(stack_path)
to_sort = []

for path in os.listdir(stack_path) :
    path = path.split('.')
    frame = path[0]
    frame = frame.split('e')
    frame[1] = int(frame[1])
    to_sort.append(frame)


frames = sorted(to_sort)
for frame in frames :
    frame_path = os.path.join(stack_path,frame[0]+'e'+str(frame[1])+'.png')
    frame = cv2.imread(frame_path)
    Stack.append(frame)

Stack_to_Video(Stack, Video_path)