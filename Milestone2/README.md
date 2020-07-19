## Re-ID of pedestrians

## Installation
The following libraries are neeeded:

	numpy == 1.17.5
	cython
	matplotlib
        pyyalm
        pandas
	torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
	
## Train
To train a new model, run the command :

	python3 train.py --gpu_ids <#> --name <name_of_model> --data_dir <training_dataset_dir> -- save_dir <save_model_dir> --PCB

additionnal options:

	--gpu_ids, default='0', 			=> gpu_ids: e.g. 0  0,1,2  0,2
	--name, default='ft_ResNet50', type=str, 	=> output model name
	--data_dir, default='../../../scratch/angomes/Milestone2/split_train',type=str, => training dir path
	--save_dir, default='../../../scratch/angomes/Milestone2/saves/model',type=str, => saving dir path, where the model will be saved
	--batchsize, default=32, type=int, 		=> batchsize
	--stride, default=2, type=int, 			=> stride
	--erasing_p, default=0, type=float, 		=> Random Erasing probability, in [0,1]
	--warm_epoch, default=0, type=int, 		=> the first K epoch that needs warm up
	--lr, default=0.05, type=float, 		=> learning rate
	--droprate, default=0.5, type=float, 		=> drop rate

By default, the backbone model will be Resnet50, for others models, add command: 
	--use_dense, 	=> use densenet121 backbone
	--PCB, 		=> use PCB+ResNet50 backbone

## Test + Evaluate
To test the model, run for example the command :

	python3 test.py --gpu_ids <#> --name <name_of_model_to_test> --data_dir <testing_dataset_dir> -- save_dir <model_dir> --PCB

additionnal options:

	--which_epoch', default='last', type=str, 	=> # of the epoch to test (if multiple epochs availalble from save): 0,1,2,3...or last
	--gpu_ids, default='0', 			=> gpu_ids: e.g. 0  0,1,2  0,2
	--name, default='ft_ResNet50', type=str, 	=> name of model test
	--test_dir, default='../../../scratch/angomes/Milestone2/split_test',type=str, 	=> testing dataset dir path
	--save_dir, default='../../../scratch/angomes/Milestone2/saves/model',type=str, => saving dir path, where the model is and where the info will be saved
	--batchsize, default=32, type=int, 		=> batchsize

If the tested model is from a different backbone than Resnet50, add commands:
	--use_dense, 	=> use densenet121 backbone
	--PCB, 		=> use PCB+ResNet50 backbone

Finally, to evaluate the model, run the command :

	python3 evaluate_gpu.py --gpu_ids <#> --name <name_of_model_to_evaluate> --save_dir <model_dir>
	
additionnal options:

	--gpu_ids, default='0', 			=> gpu_ids: e.g. 0  0,1,2  0,2
	--name, default='ft_ResNet50', type=str, 	=> name of model to evaluate
	--save_dir, default='../../../scratch/angomes/Milestone2/saves/model',type=str, => saving dir path, where the model and data to evaluate is	
