Rice [Datathon 2024](https://rice-datathon-2024.devpost.com/) NeuroTech@Rice track 1st place solution. 

[Project](https://devpost.com/software/hypnos-gvzdxo)

## How to run the code

Dependencies:`torch`, `numpy`, `pytorch_lightning`, `wandb`, `scipy`, `scikit-learn`.

Train the 1D CNN (ResNet) classification model:

```python train_classification.py --config utils/config_cls.yaml```

Train the UNet segmentation model:

```python train_segmentation.py --config utils/config_seg.yaml```


## Directories

* `assets/`: contains the figures used in the report, as well as the test set inference output;
* `ckpt/`: contains the checkpoints of the trained models;
* `models/`: contains the model definitions, in this project we use ResNet1D and UNet;
* `utils/`: contains the utility functions used in the project;
* `notebooks/`: some notebooks used for data exploration and visualization;


## Scripts


* `train_classification.py`: train the training datasets with classification, log both training result and validation result;

* `train_segmentation.py`: train the training datasets with segmentation, log both training result and validation result;

* `inference.py`: used to produce the submission file for the test set;

* `SleepDataset.py`: the dataset class for loading NeuroTech's time series data;

* Other utility functions in `utils/`, including preprocessing, mixup data augmentation, etc.

## Config Files

* `utils/config_cls.yaml`: the configuration file for training the classification model;
* `utils/config_seg.yaml`: the configuration file for training the segmentation model;


