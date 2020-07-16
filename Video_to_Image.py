import os
import cv2
import numpy as np

def Video_to_Stack(path):
    """
    Create an Image Stack from a video file:
    Inputs:
        - path: Absolute path to video
    Outputs:
        - ImageStack: Stack of images in format: (N, height, width, channels)
    """
    VideoCap = cv2.VideoCapture(path) # declare video capture object
    if (VideoCap.isOpened() == False): print('Error reading video')

    ImageStack = [] # initialize image stack
    print('\nReading Video ...', flush=True)

    while VideoCap.isOpened():
        ret, frame = VideoCap.read() 
        if ret: # if there was a frame, add it to the stack
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ImageStack.append(frame)

        else:   # Video ended
            VideoCap.release() # release the video capture object
            print('\nVideo read: {} frames'.format(len(ImageStack)), flush=True)
            break
    
    return ImageStack

def Stack_to_Video(Stack_path, Video_Path, fps):
    """
    Create Video from Image Stack in format (N, height, width, channels)
    Inputs:
        - Stack_path: folder of the Images to combine to a video
        - Video_Path: The absolute output path with file name
        - fps : number of frames per second

    Outputs:
        - None
    """
    
    # Sort the frames to the right order
    Stack = []
    to_sort = []
    
    for path in os.listdir(Stack_path) :
        path = path.split('.')
        frame = path[0]
        frame = frame.split('e')
        frame[1] = int(frame[1])
        to_sort.append(frame)
    
    
    frames = sorted(to_sort)
    for frame in frames :
        frame_path = os.path.join(Stack_path,frame[0]+'e'+str(frame[1])+'.png')
        frame = cv2.imread(frame_path)
        Stack.append(frame)
    
    # Video creation
    frame_width, frame_height = Stack[0].shape[1], Stack[0].shape[0]
    
    out = cv2.VideoWriter(Video_Path,
                # specify codec: http://www.fourcc.org/mp4v/
                #cv2.VideoWriter_fourcc(*'FFV1'), # Lossless codec
                cv2.VideoWriter_fourcc(*'DIVX'), # Lossy codec    
                fps,     # fps
                (frame_width,frame_height)) # video dimensions
    print('Writing video.', end='')
    for i, frame in enumerate(Stack):

        # Write the frame into the file 'output.avi'
        out.write(frame.astype('u1'))

        if not i%5: print('.', end='')
 
    out.release()
    print('Done!\n\n')
