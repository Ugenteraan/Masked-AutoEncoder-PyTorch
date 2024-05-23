'''This module will/can be used to load various datasets from different sources.
'''

import deeplake
import torch
from torchvision import transforms

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

