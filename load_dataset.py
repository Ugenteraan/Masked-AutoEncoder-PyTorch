'''This module will/can be used to load various datasets from different sources.
'''

import deeplake
import torch
from torchvision import transforms

class LoadDeepLakeDataset:
    '''Loads a dataset from deeplake https://datasets.activeloop.ai/docs/ml/datasets/. 
    '''

    def __init__(self, token, deeplake_ds_name, image_height, image_width, batch_size, shuffle, use_random_horizontal_flip, mode='train'):
        '''Init variables.
        '''

        self.token = token
        self.deeplake_ds_name = deeplake_ds_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.use_random_horizontal_flip = use_random_horizontal_flip
        self.image_height = image_height
        self.image_width = image_width


    def collate_fn(self, batch_data):
        '''Custom collate function to preprocess the batch dataset.
        '''
        return {
                'images': torch.stack([x['images'] for x in batch_data]),
                'labels': torch.stack([torch.from_numpy(x['labels']) for x in batch_data])
            }
    

    @staticmethod
    def training_transformation():
        '''Data augmentation for the training dataset.
        '''

        transformation_list = [
                                transforms.Resize((self.image_height, self.image_width)),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)) #to turn grayscale arrays into compatible RGB arrays.
                                ]

        
        if self.use_random_horizontal_flip:
            transformation_list.insert(1, transforms.RandomHorizontalFlip())

        return transforms.Compose(transformation_list)
    

    @staticmethod
    def testing_transformation():
        return  transforms.Compose([
            transforms.Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1))
        ])


    def __call__(self):
        
        deeplake_dataset = deeplake.load(self.deeplake_ds_name, token=self.token)


        if self.mode == 'train':
            dataloader = deeplake_dataset.dataloader().transform({'images':self.training_transformation(),
                                                                   'labels':None}).batch(self.batch_size).shuffle(self.shuffle).pytorch(collate_fn=self.collate_fn, 
                                                                                                                                        decode_method={'images':'pil'})
        else:
            dataloader = deeplake_dataset.dataloader().transform({'images':self.testing_transformation(),
                                                                  'labels':None}).batch(self.batch_size).shuffle(self.shuffle).pytorch(collate_fn=self.collate_fn,
                                                                                                                                       decode_method={'images':'pil'})

        return dataloader

