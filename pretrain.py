'''Pre-training of MAE. i.e. SSL training before any finetuning or probing.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import datetime
from loguru import logger
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
# from torch.profiler import profile, record_function, ProfilerActivity

from models.mae import MaskedAutoEncoder
from load_dataset import LoadDeepLakeDataset, LoadUnlabelledDataset
from init_optim import InitOptimWithSGDR
from utils import load_checkpoint, save_checkpoint
from visualize_prediction import VisualizePrediction
import cred

def main(args):


    DATETIME_NOW = datetime.datetime.now().replace(second=0, microsecond=0) #datetime without seconds & miliseconds.

    #Read the config file from args.
    with open(args.config, 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.FullLoader)
        print("Configuration read successful...")

    
    with open(args.logging_config, 'r') as logging_configfile:
        logging_config = yaml.load(logging_configfile, Loader=yaml.FullLoader)
        print("Logging configuration file read successful...")
        print("Initializing logger") 
    
    
    ###Logger initialization
    if logging_config['disable_default_loggers']:
        logger.remove(0)

    logging_formatter = logging_config['formatters'][config['env']] #set the environment for the logger.

    
    Path(f"{logging_config['log_dir']}").mkdir(parents=True, exist_ok=True)
    #to output to a file
    logger.add(f"{logging_config['log_dir']}{DATETIME_NOW}-{logging_config['log_filename']}",
                    level=logging_formatter['level'],
                    format=logging_formatter['format'],
                    backtrace=logging_formatter['backtrace'],
                    diagnose=logging_formatter['diagnose'],
                    enqueue=logging_formatter['enqueue'])

    #to output to the console.
    logger.add(sys.stdout,
                level=logging_formatter['level'],
                format=logging_formatter['format'],
                backtrace=logging_formatter['backtrace'],
                colorize=True,
                diagnose=logging_formatter['diagnose'],
                enqueue=logging_formatter['enqueue'])
   

    #@@@@@@@@@@@@@@@@@@@@@@@@@ Extract the configurations from YAML file @@@@@@@@@@@@@@@@@@@@@@

    #Data configurations.
    BATCH_SIZE = config['data']['batch_size']
    IMAGE_SIZE = config['data']['image_size']
    IMAGE_DEPTH = config['data']['image_depth']
    DATASET_FOLDER = config['data']['dataset_folder']
    NUM_WORKERS = config['data']['num_workers']
    SHUFFLE = config['data']['shuffle']
    USE_RANDOM_HORIZONTAL_FLIP = config['data']['use_random_horizontal_flip']
    NORMALIZE_PIXEL = config['data']['normalize_pixel']
    DEEPLAKE_DS_NAME = config['data']['deeplake_ds_name']


    #Mask configurations.
    PATCH_SIZE = config['mask']['patch_size']
    MASKING_RATIO = config['mask']['masking_ratio']

    #Visualization configurations.
    VISUALIZE_BATCH_SIZE = config['visualization']['visualize_batch_size']
    FIG_SAVEPATH = config['visualization']['fig_savepath']
    NUM_FIGS = config['visualization']['num_figs']
    VISUALIZE_FREQ = config['visualization']['visualize_freq']

    #Model configurations.
    MODEL_SAVE_FOLDER = config['model']['model_save_folder']
    MODEL_NAME = config['model']['model_name']
    MODEL_SAVE_FREQ = config['model']['model_save_freq']
    N_SAVED_MODEL_TO_KEEP = config['model']['N_saved_model_to_keep']
    ENCODER_TRANSFORMER_BLOCKS_DEPTH = config['model']['encoder_transformer_blocks_depth']
    DECODER_TRANSFORMER_BLOCKS_DEPTH = config['model']['decoder_transformer_blocks_depth']
    ENCODER_EMBEDDING_DIM = config['model']['encoder_embedding_dim']
    DECODER_EMBEDDING_DIM = config['model']['decoder_embedding_dim']
    ENCODER_MLP_RATIO = config['model']['encoder_mlp_ratio']
    DECODER_MLP_RATIO = config['model']['decoder_mlp_ratio']
    ENCODER_NUM_HEADS = config['model']['encoder_num_heads']
    DECODER_NUM_HEADS = config['model']['decoder_num_heads']
    ATTN_DROPOUT_PROB = config['model']['attn_dropout_prob']
    FEEDFORWARD_DROPOUT_PROB = config['model']['feedforward_dropout_prob']

    #Training configurations
    DEVICE = config['training']['device']
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and DEVICE=='gpu' else 'cpu')
    LOAD_CHECKPOINT = config['training']['load_checkpoint']
    LOAD_CHECKPOINT_EPOCH = config['training']['load_checkpoint_epoch']
    END_EPOCH = config['training']['end_epoch']
    START_EPOCH = config['training']['start_epoch']
    COSINE_UPPER_BOUND_LR = config['training']['cosine_upper_bound_lr']
    COSINE_LOWER_BOUND_LR = config['training']['cosine_lower_bound_lr']
    WARMUP_START_LR = config['training']['warmup_start_lr']
    WARMUP_STEPS = config['training']['warmup_steps']
    NUM_EPOCH_TO_RESTART_LR = config['training']['num_epoch_to_restart_lr']
    COSINE_UPPER_BOUND_WD = config['training']['cosine_upper_bound_wd']
    COSINE_LOWER_BOUND_WD = config['training']['cosine_lower_bound_wd']
    USE_BFLOAT16 = config['training']['use_bfloat16']
    USE_NEPTUNE = config['training']['use_neptune']    
    
    
    if USE_NEPTUNE:
        import neptune

        NEPTUNE_RUN = neptune.init_run(
                                        project=cred.NEPTUNE_PROJECT,
                                        api_token=cred.NEPTUNE_API_TOKEN
                                      )
        #we have partially unsupported types. Hence the utils method.
        NEPTUNE_RUN['parameters'] = neptune.utils.stringify_unsupported(config)


    logger.info("Init MAE model...")
    
    MAE_MODEL = MaskedAutoEncoder(patch_size=PATCH_SIZE, 
                                  image_size=IMAGE_SIZE, 
                                  image_depth=IMAGE_DEPTH,
                                  encoder_embedding_dim=ENCODER_EMBEDDING_DIM, 
                                  decoder_embedding_dim=DECODER_EMBEDDING_DIM, 
                                  encoder_transformer_blocks_depth=ENCODER_TRANSFORMER_BLOCKS_DEPTH, 
                                  decoder_transformer_blocks_depth=DECODER_TRANSFORMER_BLOCKS_DEPTH, 
                                  masking_ratio=MASKING_RATIO,
                                  normalize_pixel=NORMALIZE_PIXEL,
                                  device=DEVICE,
                                  encoder_mlp_ratio=ENCODER_MLP_RATIO, 
                                  decoder_mlp_ratio=DECODER_MLP_RATIO,
                                  encoder_num_heads=ENCODER_NUM_HEADS,
                                  decoder_num_heads=DECODER_NUM_HEADS, 
                                  attn_dropout_prob=ATTN_DROPOUT_PROB,
                                  feedforward_dropout_prob=FEEDFORWARD_DROPOUT_PROB,
                                  logger=logger).to(DEVICE, non_blocking=True)
    
    
    # DEEPLAKE_DATALOADER = LoadDeepLakeDataset(token=cred.ACTIVELOOP_TOKEN,
    #                                           deeplake_ds_name=f"hub://activeloop/{DEEPLAKE_DS_NAME}-train",
    #                                           image_size=IMAGE_SIZE,
    #                                           batch_size=BATCH_SIZE, 
    #                                           num_workers=NUM_WORKERS,
    #                                           shuffle=SHUFFLE,
    #                                           use_random_horizontal_flip=USE_RANDOM_HORIZONTAL_FLIP,
    #                                           mode='train',
    #                                           logger=logger)()


    DATASET_MODULE = LoadUnlabelledDataset(dataset_folder_path=DATASET_FOLDER, 
                                       image_size=224, 
                                       image_depth=3, 
                                       use_random_horizontal_flip=USE_RANDOM_HORIZONTAL_FLIP, 
                                       logger=logger)

    DATALOADER = DataLoader(DATASET_MODULE, 
                            batch_size=BATCH_SIZE, 
                            shuffle=SHUFFLE, 
                            num_workers=NUM_WORKERS)

    ITERATIONS_PER_EPOCH = len(DATALOADER) 
    #this module contains the init for optimizer and schedulers.
    OPTIM_AND_SCHEDULERS = InitOptimWithSGDR(
                                             autoencoder_model=MAE_MODEL,
                                             cosine_upper_bound_lr=COSINE_UPPER_BOUND_LR,
                                             cosine_lower_bound_lr=COSINE_LOWER_BOUND_LR,
                                             warmup_start_lr=WARMUP_START_LR,
                                             warmup_steps=WARMUP_STEPS,
                                             num_steps_to_restart_lr=NUM_EPOCH_TO_RESTART_LR*ITERATIONS_PER_EPOCH,
                                             cosine_upper_bound_wd=COSINE_UPPER_BOUND_WD,
                                             cosine_lower_bound_wd=COSINE_LOWER_BOUND_WD,
                                             logger=logger
                                            )

    OPTIMIZER = OPTIM_AND_SCHEDULERS.get_optimizer()
    # OPTIMIZER = torch.optim.AdamW(MAE_MODEL.parameters(), lr=1.0e-5, betas=(0.9, 0.95))
    SCALER = None

    if USE_BFLOAT16:
        SCALER = torch.cuda.amp.GradScaler()

    if LOAD_CHECKPOINT:
        MAE_MODEL, START_EPOCH = load_checkpoint(model_save_folder=MODEL_SAVE_FOLDER, 
                                                                    model_name=MODEL_NAME, 
                                                                    mae_model=MAE_MODEL,
                                                                    load_checkpoint_epoch=None, 
                                                                    logger=logger)
        for _ in range(ITERATIONS_PER_EPOCH*START_EPOCH):
            OPTIM_AND_SCHEDULERS.step() #this is needed to restore the parameters of the optimizer. LR and WD rate.


    #Initialize the visualizing module to visualize the predictions.
    VISUALIZER = VisualizePrediction(visualize_batch_size=VISUALIZE_BATCH_SIZE, 
                                     image_size=IMAGE_SIZE,
                                     image_depth=IMAGE_DEPTH,
                                     patch_size=PATCH_SIZE,
                                     fig_savepath=FIG_SAVEPATH,
                                     num_figs=NUM_FIGS)


    
    for epoch_idx in range(START_EPOCH, END_EPOCH):

        logger.info(f"Training has started for epoch {epoch_idx}")
        
        MAE_MODEL.train() #set to train mode.

        epoch_loss = 0
        

        for idx, data in tqdm(enumerate(DATALOADER)):
            
            OPTIMIZER.zero_grad()

            images = data['images'].to(DEVICE)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=USE_BFLOAT16):
                loss, preds, inverted_masks = MAE_MODEL(x=images)
                
            
            # SCALER.scale(loss).backward()
            # OPTIMIZER.step()
            #backward and step
            if USE_BFLOAT16:
                SCALER.scale(loss).backward()
                SCALER.step(OPTIMIZER)
                SCALER.update()
            else:
                loss.backward()
                OPTIMIZER.step()
            
            _new_lr, _new_wd = OPTIM_AND_SCHEDULERS.step()
            epoch_loss += loss.item()

            # VISUALIZER.plot(pred_tensor=preds.detach(), 
            #                 target_tensor=images.detach(), 
            #                 inverted_masks=inverted_masks.detach(),
            #                 epoch_idx=epoch_idx)

        
        
        logger.info(f"The training loss at epoch {epoch_idx} is : {epoch_loss}")
        
        if USE_NEPTUNE:
            NEPTUNE_RUN['train/loss_per_epoch'].append(epoch_loss)
            
        

        if epoch_idx % VISUALIZE_FREQ == 0:

            #visualize the last iteration.
            VISUALIZER.plot(pred_tensor=preds.detach(), 
                            target_tensor=images.detach(), 
                            inverted_masks=inverted_masks.detach(),
                            epoch_idx=epoch_idx)

        
        if epoch_idx % MODEL_SAVE_FREQ == 0:
            
            save_checkpoint(model_save_folder=MODEL_SAVE_FOLDER, 
                    model_name=MODEL_NAME, 
                    mae_model=MAE_MODEL, 
                    scaler=SCALER, 
                    epoch=epoch_idx, 
                    loss=epoch_loss, 
                    N_models_to_keep=N_SAVED_MODEL_TO_KEEP, 
                    logger=logger
                    )
        
    if USE_NEPTUNE:
        NEPTUNE_RUN.stop() 
                                     
                                  

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Specify the YAML config file to be used.')
    parser.add_argument('--logging_config', required=True, type=str, help='Specify the YAML config file to be used for the logging module.')
    args = parser.parse_args()
    main(args)
