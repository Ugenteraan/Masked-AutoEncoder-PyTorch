'''Pre-training of MAE. i.e. SSL training before any finetuning or probing.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import datetime
import loguru

import torch.nn as nn


def main():

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


    #Mask configurations.
    PATCH_SIZE = config['mask']['patch_size']
    MASKING_RATIO = config['mask']['masking_ratio']

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


    logger.info("Init MAE model...")
    
    MAE_MODEL = MaskedAutoEncoder( 
                                     patch_size=PATCH_SIZE, 
                                     image_size=IMAGE_SIZE, 
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
                                     feedforward_dropout_prob=FEEDFORWARD_DROPOUT_PROB)
                                     
                                  
                                     
                                     










if __name__ == '__main__':

    main()
