from ECPdataset import ECPdataset
from get_resnet50_fpn import get_resnet50_fpn
from get_detection_model import get_detection_model
from engine import train_one_epoch, evaluate, get_transform
import torch
import utils
import argparse 

"""
To run in the console, input commands:
    
    python run_train.py -s 'saves/resnet50_fullytrained_8bst_lr0dot001_ss5_gamma0dot3' 
    -m 'saves/resnet50_fullytrained_8bst_lr0dot001_ss5_gamma0dot3'
    
with:
    
-s the path where to save to trained model and
-m the path from where to load a pretrained model to further train
    
"""

def train_model(path_to_save='saves/resnet50_fullytrained_8bst_lr0dot001_ss5_gamma0dot3', path_to_model=None):
    
    if path_to_save == None:
        path_to_save = path_to_save='saves/resnet50_fullytrained_8bst_lr0dot001_ss5_gamma0dot3'
        
    # DATASET
    # use dataset and defined transformations
    dataset_train = ECPdataset('../ECP/day/','train', get_transform(train=True))
    dataset_valid = ECPdataset('../ECP/day/','val', get_transform(train=False))
    
    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
    
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    
    # MODEL
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Three classes - background, pedestrian, rider
    num_classes = 3
    
    # Get the model using helper function
    model = get_resnet50_fpn(num_classes)
    #model = get_detection_model(num_classes, model='mobilenet', anch='15-highersize')
    model.to(device)
    
    # OPTIMIZER
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005)
        # Learning rate scheduler which decreases the learning rate by 3x every 5 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5, gamma=0.3)
    
    # LOAD SAVE
    if path_to_model != None:
        checkpoint = torch.load(path_to_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epochs']
        history = checkpoint['history']
        model.epochs = epoch
        best_epoch = epoch

    model.train()   
    
    # INITIALISATION
    num_epochs = 50
    
    # Early stopping intialization
    epochs_no_improve = 0
    max_epochs_stop = 10
    
    try:
        valid_AP_max = checkpoint['valid_AP_max']
        print(f'Model has been trained for: {model.epochs} epochs, with {valid_AP_max:.4f} validation AP.\nStarting training...', \
              flush=True)

    except:
        model.epochs = 0
        valid_AP_max = 0
        history = []
        print(f'Starting training from scratch.', flush=True)
    
    # TRAIN
    for epoch in range(model.epochs+1, num_epochs+1):
        # train for one epoch, printing every 10 iterations
        train_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
    
        # update the learning rate
        lr_scheduler.step()
    
        # evaluate on the test dataset
        metrics = evaluate(model, data_loader_valid, device=device)
        valid_AP = metrics[0]
    
        # save info
        history.append([train_loss, valid_AP])
        print(
              f'\nTraining loss: {train_loss:.4f} and AP: {valid_AP:.4f}',
              flush=True
                )
        
        # Save the model if validation AP increases
        if valid_AP > valid_AP_max:
            print('Saving model',flush=True)
            torch.save({
                'epochs': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'valid_AP_max': valid_AP,
                'history': history
                }, path_to_save)
            # Track improvement
            epochs_no_improve = 0
            valid_AP_max = valid_AP
            best_epoch = epoch
    
        # Otherwise increment count of epochs with no improvement
        else:
            epochs_no_improve += 1
            # Trigger early stopping
            if epochs_no_improve >= max_epochs_stop:
                print(
                    f'\nEarly stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with AP: {valid_AP_max:.4f}', \
                flush=True)
                break
        
    print(f'\nFinished training! Total epochs: {epoch}. Best epoch: {best_epoch} with AP: {valid_AP_max:.4f}', \
          flush=True)
    return    
        
def main(path_to_save='saves/resnet50_fullytrained_8bst_lr0dot001_ss5_gamma0dot3', path_to_model=None):
    prediction = train_model(path_to_save, path_to_model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--path_save", required=False, help="path to save")
    ap.add_argument("-m", "--path_model", required=False, help="path to model")

    args = vars(ap.parse_args())
    main(args["path_save"],args["path_model"])
    