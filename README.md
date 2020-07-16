# Pedestrians identification and tracking 
### Students : André Gomes, Yann Martinson, Sylvain Pietropaolo

This final project (Milestone 3) combines two previous projects where models where trained for pedestrian detection (Milestone 1) and re-identification (Milestone 2). The objective is here to take a video as input and output the same video with labelled bounding boxes on the different pedestrians tracked.

## Examples
Three examples of the work done by the proposed algorithm are found in folder Final_videos. 

## Installation
The libraries in requirements.txt are neeeded to properly execute the code. 

## GPU or not
The code is written in order to run without GPU. To aceelerate the execution, small changes
should be done in the code.

## Get trained models
Identification: the trained model (resnet50_ped) on ECP dataset can be found here:

https://drive.google.com/file/d/1xofm9jQtkgiIuO3XynwKVB1fbuVsXpPg/view?usp=sharing

Re-identification: the trained model (PCB.pth) on the MSMT17 dataset can be found here:

https://drive.google.com/file/d/1eERK2rQ5E84i_thb-rhfkNY-7rX0zsxe/view?usp=sharing

Both model files should be saved inside the 'Models' folder.

## Execution
To execute the code, run the following command:

	python3 run.py --path_to_video <path_to_video> --path_to_stack <path_to_stack>  --path_to_save <path_to_save> --show True

options:

	--path_to_video 	default='./Videos/MOT16-10-raw.webm'		=> path to the input video
	--path_to_stack 	default='./images/new_frames/video1'	 	=> path to the output images stack
	--path_to_save	 	default='./Final_videos/final_video9.avi'	=> path to the output video
	--show		 	default=False					=> show each frame after processing

