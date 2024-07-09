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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
# from torch.profiler import profile, record_function, ProfilerActivity

from models.finetune_model import PretrainedEncoder, FineTuneModelClassification
from load_dataset import LoadLabelledDataset
from utils import calculate_accuracy, load_encoder_checkpoint
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
    LOAD_PRETRAINED = config['model']['load_pretrained']
    PRETRAIN_MODEL_SAVE_FOLDER = config['model']['pretrain_model_save_folder']
    PRETRAIN_MODEL_NAME = config['model']['pretrain_model_name']
    MODEL_SAVE_FOLDER = config['model']['model_save_folder']
    MODEL_NAME = config['model']['model_name']
    MODEL_SAVE_FREQ = config['model']['model_save_freq']
    N_SAVED_MODEL_TO_KEEP = config['model']['N_saved_model_to_keep']
    PATCH_SIZE = config['model']['patch_size']
    ENCODER_TRANSFORMER_BLOCKS_DEPTH = config['model']['encoder_transformer_blocks_depth']
    ENCODER_EMBEDDING_DIM = config['model']['encoder_embedding_dim']
    ENCODER_MLP_RATIO = config['model']['encoder_mlp_ratio']
    ENCODER_NUM_HEADS = config['model']['encoder_num_heads']
    ATTN_DROPOUT_PROB = config['model']['attn_dropout_prob']
    FEEDFORWARD_DROPOUT_PROB = config['model']['feedforward_dropout_prob']
    CLASSIFICATION_EXPANSION_FACTOR = config['model']['classification_expansion_factor']


    #Training configurations
    DEVICE = config['training']['device']
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and DEVICE=='gpu' else 'cpu')
    LOAD_CHECKPOINT = config['training']['load_checkpoint']
    LOAD_CHECKPOINT_EPOCH = config['training']['load_checkpoint_epoch']
    END_EPOCH = config['training']['end_epoch']
    START_EPOCH = config['training']['start_epoch']
    USE_BFLOAT16 = config['training']['use_bfloat16']
    LEARNING_RATE = config['training']['learning_rate']
    USE_NEPTUNE = config['training']['use_neptune']
    USE_TENSORBOARD = config['training']['use_tensorboard']
    USE_PROFILER = config['training']['use_profiler']



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
                                         feedforward_dropout_prob=FEEDFORWARD_DROPOUT_PROB,
                                         logger=None).to(DEVICE)



    #load the pretrained model weights.
    if LOAD_PRETRAINED:
        ENCODER_NETWORK, _ = load_encoder_checkpoint(model_save_folder=PRETRAIN_MODEL_SAVE_FOLDER, 
                                                  mae_model_name=PRETRAIN_MODEL_NAME, 
                                                  encoder_model=ENCODER_NETWORK, 
                                                  load_checkpoint_epoch=None, 
                                                  logger=logger)

    
    

    CLASSIFICATION_NETWORK = FineTuneModelClassification(input_dim=ENCODER_EMBEDDING_DIM,
                                                         expansion_factor=CLASSIFICATION_EXPANSION_FACTOR,
                                                         num_class=NUM_CLASS,
                                                         device=DEVICE, 
                                                         logger=None).to(DEVICE)


      
    TRAIN_DATASET_MODULE = LoadLabelledDataset(dataset_folder_path=DATASET_FOLDER)
    TEST_DATASET_MODULE = LoadLabelledDataset(dataset_folder_path=DATASET_FOLDER, train=False)
    

    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET_MODULE, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=SHUFFLE, 
                                  num_workers=NUM_WORKERS)


    TEST_DATALOADER = DataLoader(TEST_DATASET_MODULE, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=False, 
                                  num_workers=NUM_WORKERS)

    
    OPTIMIZER = torch.optim.AdamW(CLASSIFICATION_NETWORK.parameters(), lr=LEARNING_RATE)
    CRITERION = torch.nn.CrossEntropyLoss().to(DEVICE)

    
    SCALER = None

    #scaler is used to scale the values in variables like state_dict, optimizer etc to bfloat16 type.
    if USE_BFLOAT16:
        SCALER = torch.cuda.amp.GradScaler()
    
    
    ENCODER_NETWORK.eval()

    for epoch_idx in range(START_EPOCH, END_EPOCH):

        logger.info(f"Training has started for epoch {epoch_idx}")

        
        CLASSIFICATION_NETWORK.train()

        train_running_loss = 0
        train_running_accuracy= 0
        train_idx = 0

        for train_idx, data in enumerate(TRAIN_DATALOADER):

            OPTIMIZER.zero_grad() 

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=USE_BFLOAT16):

                batch_x, batch_y =  data['images'].to(DEVICE), data['labels'].to(DEVICE)

                feature_tensor = ENCODER_NETWORK(batch_x)
                prediction = CLASSIFICATION_NETWORK(feature_tensor)

            
            batch_loss = CRITERION(input=prediction, target=batch_y)
            
            train_running_loss += batch_loss.item()
            train_running_accuracy += calculate_accuracy(predicted=prediction.detach(), target=batch_y)

            #backward and step
            if USE_BFLOAT16:
                SCALER.scale(batch_loss).backward()
                SCALER.step(OPTIMIZER)
                SCALER.update()
            else:
                batch_loss.backward()
                OPTIMIZER.step()

        
        train_total_loss = train_running_loss
        train_total_accuracy = train_running_accuracy/(train_idx+1)

        logger.info(f"Total train loss at epoch {epoch_idx} is {train_total_loss}")
        logger.info(f"Total train accuracy at epoch {epoch_idx} is {train_total_accuracy}")
        
        
        test_idx = 0
        test_running_loss = 0
        test_running_accuracy = 0

        for test_idx, data in enumerate(TEST_DATALOADER):
            
            CLASSIFICATION_NETWORK.eval()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=USE_BFLOAT16):
                batch_x, batch_y =  data['images'].to(DEVICE), data['labels'].to(DEVICE)

                feature_tensor = ENCODER_NETWORK(batch_x)
                prediction = CLASSIFICATION_NETWORK(feature_tensor)

            batch_loss = CRITERION(input=prediction, target=batch_y)
            
            test_running_loss += batch_loss.item()
            test_running_accuracy += calculate_accuracy(predicted=prediction.detach(), target=batch_y)

        test_total_loss = test_running_loss
        test_total_accuracy = test_running_accuracy/(test_idx+1)

        logger.info(f"Total test loss at epoch {epoch_idx} is {test_total_loss}")
        logger.info(f"Total test accuracy at epoch {epoch_idx} is {test_total_accuracy}")


        
        if USE_NEPTUNE:
            NEPTUNE_RUN['train/loss_per_epoch'].append(train_total_loss)
            NEPTUNE_RUN['train/accuracy_per_epoch'].append(train_total_accuracy)
            NEPTUNE_RUN['test/loss_per_epoch'].append(test_total_loss)
            NEPTUNE_RUN['test/accuracy_per_epoch'].append(test_total_accuracy)

        
        if USE_TENSORBOARD:
            TB_WRITER.add_scalar("Loss/train", train_total_loss , epoch_idx)
            TB_WRITER.add_scalar("Accuracy/train", train_total_accuracy, epoch_idx)
            TB_WRITER.add_scalar("Loss/test", test_total_loss, epoch_idx)
            TB_WRITER.add_scalar("Accuracy/train", test_total_accuracy, epoch_idx)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Specify the YAML config file to be used.')
    parser.add_argument('--logging_config', required=True, type=str, help='Specify the YAML config file to be used for the logging module.')
    args = parser.parse_args()
    main(args)
