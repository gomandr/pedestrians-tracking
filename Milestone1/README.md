## Faster R-CNN for pedestrians detection

### Credits
Based on existing Pytorch tutorial available under: https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb.

### Installation
Requires the following libraries:

	numpy == 1.17.5
	cython
	matplotlib
	torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
	git + https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI

Run the command : pip install -r requirements.txt

Additionally, the folder "Milestone1" should be in the same directory as the ECP dataset, if that is not the case one has to do minor changes in the "run_train.py" file to make it work.

### Predict
To see a prediction on a specified image of the validation set, orun the command : 

	python run_predict.py -i path_to_image -m path_to_model

With "path_to_image" the path to the image, and "path_to_model" the path to the model used for the prediction.

This will open the images with the bounding boxes and will return the coordinates of the boxes with the labels.

### Train
To train a new model one should run the command :

	python run_train.py -s path_to_save -m path_to_model

With "path_to_save" the location where to save the new trained model and "path_to_model" the path to the model from which
the training begin. If the variable "path_to_model" is not specified the model is trained from scratch.

This will train the specified model and return the current epoch and the metrics at the end of each epoch until the training stops.

- By default the model will be trained with a full pretrained (on the COCO dataset) Resnet50 backbone, an initial learning rate of 0.01,
  15 anchors with large size (the larger between the 2 proposed inside the code), a stopping criterion of 5 epochs without improvements and an SGD
  optimizer. To change these parameters, modify "run_train.py" accordingly.
- Don't forget to change the model name "path_to_save" in order to not delete the previous trained model.
- Backbone/anchors modification : Use the function "get_model()" with the wanted variables instead of the "get_resnet50_fpn" by
				  commenting and uncommenting the right lines.
 
