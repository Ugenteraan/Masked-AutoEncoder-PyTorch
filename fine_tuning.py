'''We perform another round of training with labelled dataset by leveraging the pretrained models.
The idea here is to:
    1) Prepare a relatively small labelled dataset for finetune training.
    2) Build a transformer model that are identical to the encoder of the trained MAE's encoder.
    3) Add extra (few) layer(s) at the back of the model to perform the downstream task.
    4) Load the model.
    5) Make sure the layers of the encoder are mostly frozen. In this script, we're allowing the last few layers of the trained encoder to fine-tune their weights.
    6) Train this newly initialized model.
    7) Perform evaluation.
    8) Optionally, we will also run another evaluation without the pre-trained model's weights. This is to see the effectiveness of the trained model.
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from loguru import logger
import argparse
import yaml

from models.finetune_model import PretrainedEncoder, FineTuneModelClassification



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
    NUM_CLASS = config['data']['num_class']

     
    #Model configurations.
    MODEL_SAVE_FOLDER = config['model']['model_save_folder']
    MODEL_NAME = config['model']['model_name']
    MODEL_SAVE_FREQ = config['model']['model_save_freq']
    N_SAVED_MODEL_TO_KEEP = config['model']['N_saved_model_to_keep']
    ENCODER_TRANSFORMER_BLOCKS_DEPTH = config['model']['encoder_transformer_blocks_depth']
    ENCODER_EMBEDDING_DIM = config['model']['encoder_embedding_dim']
    ENCODER_MLP_RATIO = config['model']['encoder_mlp_ratio']
    ENCODER_NUM_HEADS = config['model']['encoder_num_heads']
    ATTN_DROPOUT_PROB = config['model']['attn_dropout_prob']
    CLASSIFICATION_EXPANSION_FACTOR = config['model']['classification_expansion_factor']

    #Mask configurations.
    PATCH_SIZE = config['mask']['patch_size']


    #Training configurations
    DEVICE = config['training']['device']
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and DEVICE=='gpu' else 'cpu')
    LOAD_CHECKPOINT = config['training']['load_checkpoint']
    LOAD_CHECKPOINT_EPOCH = config['training']['load_checkpoint_epoch']



    NEPTUNE_RUN=None
    if USE_NEPTUNE:
        import neptune

        NEPTUNE_RUN = neptune.init_run(
                                        project=cred.NEPTUNE_PROJECT,
                                        api_token=cred.NEPTUNE_API_TOKEN
                                      )
        #we have partially unsupported types. Hence the utils method.
        NEPTUNE_RUN['parameters'] = neptune.utils.stringify_unsupported(config)


    if USE_TENSORBOARD:
        TB_WRITER = writer = SummaryWriter(f'runs/mae-{DATETIME_NOW}')

    logger.info("Init MAE model...")
    

    ENCODER_NETWORK = PretrainedEncoder(patch_size=PATCH_SIZE, 
                                         image_size=IMAGE_SIZE, 
                                         image_depth=IMAGE_DEPTH,
                                         encoder_embedding_dim=ENCODER_EMBEDDING_DIM, 
                                         encoder_transformer_blocks_depth=ENCODER_TRANSFORMER_BLOCKS_DEPTH, 
                                         device=DEVICE,
                                         encoder_num_heads=ENCODER_NUM_HEADS,
                                         encoder_mlp_ratio=ENCODER_MLP_RATIO,
                                         attn_dropout_prob=ATTN_DROPOUT_PROB,
                                         feedforward_dropout_prob,
                                         logger=None)
    
    

    CLASSIFICATION_NETWORK = FineTuneModelClassification(input_dim=ENCODER_EMBEDDING_DIM,
                                                         expansion_factor=CLASSIFICATION_EXPANSION_FACTOR,
                                                         num_class=NUM_CLASS,
                                                         device=DEVICE, 
                                                         logger=None)


      
    TRAIN_DATASET_MODULE = LoadLabelledDataset(dataset_folder_path=DATASET_FOLDER)
    TEST_DATASET_MODULE = LoadUnlabelledDataset(dataset_folder_path=DATASET_FOLDER, train=False)
    
     


    

   

    





if __name__ == '__main__':
    main()
