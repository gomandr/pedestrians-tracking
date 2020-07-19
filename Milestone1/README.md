### Faster R-CNN

## Installation
One should have the following libraries :
	- numpy == 1.17.5
	- cython
	- matplotlib
	- torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
	- git + https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI

Run the command : pip install -r requirements.txt

One should have the folder "Milestone1" at the same root of the ECP dataset, if that is not the case one has to do minor changes in the "run_train.py" file
to make it work.

## Get pretrained model
If repository downloaded from Moodle, due to the 50Mo limit no pretrained model will be inside the '\saves' directory. 
A trained model on the Amsterdam subset of ECP dataset can be found here (requires EPFL account):
https://drive.google.com/open?id=1k8MjvaBTX2J4lO-8dNJEvws03oknk7t7

## Predict
To see a prediction on a specified image of the validation set one should run for example the command : 

	python run_predict.py -i path_to_image -m path_to_model

With "path_to_image" the path to the image, and "path_to_model" the path to the model used for the prediction.
If the variable "path_to_model" is not not specified, the model used by default is the best one. that we trained and which is provided in the "saves" directory.

This will open the images with the bounding boxes and will return the coordinates of the boxes with the labels.

## Train
To train a new model one should run the command :

	python run_train.py -s path_to_save -m path_to_model

With "path_to_save" the location where to save the new trained model and "path_to_model" the path to the model from which
the training begin. If the variable "path_to_model" is not specified the model is trained from scratch.

This will train the specified model and return the current epoch and the metrics at the end of each epoch until the training stops.

- By default the model will be trained with a full pretrained (on the COCO dataset) Resnet50 backbone, an initial learning rate of 0.01,
  15 anchors with large size (the larger between the 2 proposed inside the code), a stopping criterion of 5 epochs without improvements and an SGD
  optimizer. To change these parameters, modify "run_train.py" accordingly.
- Don't forget to change the model name "path_to_save" in order to don't delete the previous existant model.
- The new model is saved under "path_to_save". The folder "saves" is the directory where some pretrained models are already saved.
- Backbone/anchors modification : Use the function "get_model()" with the wanted variables instead of the "get_resnet50_fpn" by
				  commenting and uncommenting the right lines.
 
