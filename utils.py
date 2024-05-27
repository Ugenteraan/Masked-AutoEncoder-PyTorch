'''Helper functions.
'''
import os
import glob
import torch
from pathlib import Path

def load_checkpoint(model_save_folder, 
                    model_name, 
                    mae_model, 
                    load_checkpoint_epoch=None, 
                    logger=None):
    '''Loads either the latest model (if load_checkpoint_val is None) or loads the specific checkpoint.
    '''

    try:
        checkpoint = None 
        if not load_checkpoint_epoch is None:
            checkpoint = torch.load(f"{model_save_folder.rstrip('/')}/{model_name}-checkpoint-ep-{load_checkpoint_epoch}.pth.tar")
        else:
            checkpoint = torch.load(f"{model_save_folder.rstrip('/')}/{model_name}-latest.pth.tar")

        epoch = checkpoint['epoch']
        logger.info(f"Checkpoint from epoch {epoch} is successfully loaded! Extracting the parameters to load to individual model/variabels now...")



    
    except Exception as err:
        logger.error(f"Error loading the model! {err}")
        epoch = 0

    return mae_model, epoch
        



def save_checkpoint(model_save_folder, 
                    model_name, 
                    mae_model, 
                    scaler, 
                    epoch, 
                    loss, 
                    N_models_to_keep, 
                    logger=None,
                    ):
    '''Save model checkpoint.
    '''
    save_dict = {
                'mae_model': mae_model.state_dict(),
                'scaler': scaler, 
                'epoch': epoch, #useful for resuming training from the last epoch. And also to initialize the optimizer module's step.
                'loss' : loss #record purposes. 
                }
    
    try:
        Path(f"{model_save_folder}").mkdir(parents=True, exist_ok=True) #create directory if doesn't exist.yy
        torch.save(save_dict, f"{model_save_folder.rstrip('/')}/{model_name}-checkpoint-ep-{epoch}.pth.tar") 
        torch.save(save_dict, f"{model_save_folder.rstrip('/')}/{model_name}-latest.pth.tar") 
        logger.info(f"Model checkpoint save for epoch {epoch} is successful!")

        #remove the unwanted models.
        remove_old_models(N_models_to_keep=N_models_to_keep, model_save_folder=model_save_folder)

    except Exception as err:
        logger.error(f"Model checkpoint save for epoch {epoch} has failed! {err}")

    return None






def remove_old_models(N_models_to_keep, model_save_folder):
    '''Remove the old saved models based on the given paramters.
    '''

    all_models = []
    for x in glob.glob(f'{model_save_folder.rstrip("/")}/**'):
        all_models.append(x)


    if len(all_models) > N_models_to_keep:
        all_models.sort(key=lambda x: os.path.getctime(x)) #sorts the files based on their creation time.
        unwanted_models = all_models[:-1*N_models_to_keep]

        if len(unwanted_models) != 0:
            #delete the old models.
            for x in unwanted_models:
                os.remove(x)

    return None
