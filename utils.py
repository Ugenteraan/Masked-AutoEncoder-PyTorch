'''Helper functions.
'''

def load_checkpoint(model_save_folder, 
                    model_name, 
                    mae_model, 
                    optimizer, 
                    scaler, 
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
        logger.info("Checkpoint from epoch {epoch} is successfully loaded! Extracting the parameters to load to individual model/variabels now...")

        optimizer.load_state_dict(checkpoint['optimizer'])

        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])    

        logger.info(f"Loaded optimizers and scalers from checkpoint...")

    
    except Exception as err:
        logger.error(f"Error loading the model! {err}")
        epoch = 0

    return mae_model, optimizer, scaler, epoch
        



def save_checkpoint(model_save_folder, 
                    model_name, 
                    mae_model, 
                    optimizer, 
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
                'optimizer': optimizer,
                'scaler': scaler, 
                'epoch': epoch, #useful for resuming training from the last epoch.
                'loss' : loss #record purposes. 
                }
    
    try:
        torch.save(save_dict, f"{model_save_folder.rstrip('/')}/{model_name}-checkpoint-ep-{epoch}.pth.tar") 
        torch.save(save_dict, f"{model_save_folder.rstrip('/')}/{model_name}-latest.pth.tar") 
        logger.info(f"Model checkpoint save for epoch {epoch} is successful!")

        #remove the unwanted models.
        remove_old_models(N_models_to_keep=N_models_to_keep, model_save_folder=model_save_folder)

    except Exception as err:
        logger.error(f"Model checkpoint save for epoch {epoch} has failed! {err}")

    return None


def calculate_accuracy(predicted, target):
    '''Calculates the accuracy of the prediction.
    '''


    num_data = target.size()[0]
    predicted = torch.argmax(predicted, dim=1)

    correct_pred = torch.sum(predicted == target)

    accuracy = correct_pred*(num_data/100)

    return accuracy.item()



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