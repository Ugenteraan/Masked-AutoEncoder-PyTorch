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

        mae_model.load_state_dict(checkpoint['mae_model']) #load the weights into the model
        epoch = checkpoint['epoch']
        if not logger is None:
            logger.info(f"Checkpoint from epoch {epoch} is successfully loaded! Extracting the parameters to load to individual model/variabels now...")



    
    except Exception as err:
        if not logger is None:
            logger.error(f"Error loading the model! {err}")
        else:
            print(err)
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

        if not logger is None:
            logger.info(f"Model checkpoint save for epoch {epoch} is successful!")

        #remove the unwanted models.
        remove_old_models(N_models_to_keep=N_models_to_keep, model_save_folder=model_save_folder)

    except Exception as err:
        if not logger is None:
            logger.error(f"Model checkpoint save for epoch {epoch} has failed! {err}")
        else:
            print(err)

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



def load_encoder_checkpoint(model_save_folder, 
                            mae_model_name, 
                            encoder_model, 
                            load_checkpoint_epoch=None, 
                            logger=None):
    '''Loads only the encoder part of the MAE network for downstream purposes.
    '''

    try:
        checkpoint = None 
        if not load_checkpoint_epoch is None:
            checkpoint = torch.load(f"{model_save_folder.rstrip('/')}/{mae_model_name}-checkpoint-ep-{load_checkpoint_epoch}.pth.tar")
        else:
            checkpoint = torch.load(f"{model_save_folder.rstrip('/')}/{mae_model_name}-latest.pth.tar")


        #to only load the patch embedding layers and the encoder transformer blocks into the new encoder model.
        filtered_state_dict = {}
        for k, v in checkpoint['mae_model'].items():
            
            if 'encoder_transformer_blocks' in k or 'patch_embed' in k or 'encoder_norm' in k:
                filtered_state_dict[k[7:]] = v #for some reason the keys here starts with 'module' while the new encoder model does not. So the 7: is to remove the 'module'.


        encoder_model.load_state_dict(filtered_state_dict, strict=True) #load the weights into the model
        epoch = checkpoint['epoch']
        if not logger is None:
            logger.info(f"Checkpoint from epoch {epoch} is successfully loaded! Extracting the parameters to load to individual model/variabels now...")

    
    except Exception as err:
        if not logger is None:
            logger.error(f"Error loading the model! {err}")
        else:
            print(err)
        epoch = 0

    return encoder_model, epoch




def calculate_accuracy(predicted, target):
    '''Calculates the accuracy of the prediction.
    '''

    num_data = target.size()[0]
    predicted = torch.argmax(predicted, dim=1)

    correct_pred = torch.sum(predicted == target)

    accuracy = (correct_pred/num_data)*100

    return accuracy.item()






