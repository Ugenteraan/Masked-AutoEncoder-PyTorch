'''This module will/can be used to load various datasets from different sources.
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os 
import glob
from PIL import Image
import numpy as np
import deeplake
import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class LoadDeepLakeDataset:
    '''Loads a dataset from deeplake https://datasets.activeloop.ai/docs/ml/datasets/. 
    '''

    def __init__(self, 
                 token, 
                 deeplake_ds_name, 
                 image_size, 
                 batch_size, 
                 num_workers,
                 shuffle, 
                 use_random_horizontal_flip, 
                 mode='train', 
                 logger=None):
        '''Init variables.
        '''

        self.token = token
        self.deeplake_ds_name = deeplake_ds_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.mode = mode
        self.use_random_horizontal_flip = use_random_horizontal_flip
        self.image_size = image_size 


    def collate_fn(self, batch_data):
        '''Custom collate function to preprocess the batch dataset.
        '''
        return {
                'images': torch.stack([x['images'] for x in batch_data]),
                'labels': torch.stack([torch.from_numpy(x['labels']) for x in batch_data])
            }
    

    def training_transformation(self):
        '''Data augmentation for the training dataset.
        '''

        transformation_list = [
                                transforms.Resize((self.image_size, self.image_size)),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)), #to turn grayscale arrays into compatible RGB arrays.
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]

        
        if self.use_random_horizontal_flip:
            transformation_list.insert(0, transforms.RandomHorizontalFlip())
        
        return transforms.Compose(transformation_list)
    

    def testing_transformation(self):
        return  transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __call__(self):
        
        deeplake_dataset = deeplake.load(self.deeplake_ds_name, token=self.token)


        if self.mode == 'train':
            dataloader = deeplake_dataset.dataloader().transform({'images':self.training_transformation(),
                                                                   'labels':None}).batch(self.batch_size).shuffle(self.shuffle).pytorch(num_workers=self.num_workers,
                                                                                                                                        collate_fn=self.collate_fn, 
                                                                                                                                        decode_method={'images':'pil'})
        else:
            dataloader = deeplake_dataset.dataloader().transform({'images':self.testing_transformation(),
                                                                  'labels':None}).batch(self.batch_size).shuffle(self.shuffle).pytorch(collate_fn=self.collate_fn,
                                                                                                                                       num_workers=self.num_workers,
                                                                                                                                       decode_method={'images':'pil'})

        return dataloader



class LoadUnlabelledDataset(Dataset):
    '''Loads the dataset from the given path.
    '''

    def __init__(self, dataset_folder_path, image_size=224, image_depth=3, use_random_horizontal_flip=False, logger=None):
        '''Parameter Init.
        '''

        if dataset_folder_path is None:
            logger.error("Dataset folder path must be provided!")
            sys.exit()

        self.dataset_folder_path = dataset_folder_path
        self.image_size = image_size
        self.image_depth = image_depth
        self.image_path = self.read_folder()
        self.logger = logger

        transformation_list = [
                                transforms.Resize((self.image_size, self.image_size)),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)), #to turn grayscale arrays into compatible RGB arrays.
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]

        
        if use_random_horizontal_flip:
            transformation_list.insert(0, transforms.RandomHorizontalFlip())

        self.transform = transforms.Compose(transformation_list)



    def read_folder(self):
        '''Reads the folder for the images.
        '''
        
        image_path = []
    
        folder_path = f"{self.dataset_folder_path.rstrip('/')}/"

        for x in glob.glob(folder_path + "**", recursive=True):

            if not x.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            image_path.append(x)

        return image_path


    def __len__(self):
        '''Returns the total size of the data.
        '''
        return len(self.image_path)

    def __getitem__(self, idx):
        '''Returns a single image and its corresponding label.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_path[idx]


        try:

            image = Image.open(image_path).convert('RGB')

        except Exception as err:
            if self.logger is not None:
                self.logger.error(f"{image_path}")
                self.logger.error(f"Error loading image: {err}")
            sys.exit()


        if self.transform:
            image = self.transform(image)

        return {
            'images': image
        }



# if __name__ == '__main__':

     
#     DATASET_MODULE = LoadUnlabelledDataset(dataset_folder_path='./dog_breed_classification/ssl_train/', 
#                                        image_size=224, 
#                                        image_depth=3, 
#                                        use_random_horizontal_flip=True, 
#                                        logger=None)

#     DATALOADER = DataLoader(DATASET_MODULE, 
#                             batch_size=64, 
#                             shuffle=True, 
#                             num_workers=8,
#                             pin_memory=True)


#     for idx, data in enumerate(DATALOADER):

#         images = data['images'].to(torch.device("cuda:0"))

#         print(idx)




