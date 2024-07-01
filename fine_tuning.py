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


def main(args=None):
    


    TRAIN_DATASET_MODULE = LoadLabelledDataset(dataset_folder_path=FINETUNE_DATASET_PATH
   

    





if __name__ == '__main__':
    main()
